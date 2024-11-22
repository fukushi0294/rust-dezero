use crate::core::function::{self, BiFunction, Function, UniFunction};
use crate::enable_backprop;
use ndarray::{iter, Array, Array1, IxDyn};
use ndarray::{Dim, IxDynImpl};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::ptr;
use std::rc::Rc;

pub struct VarData {
    pub data: Array<f64, IxDyn>,
}

impl VarData {
    pub fn new(data: Array<f64, IxDyn>) -> Self {
        return VarData { data: data };
    }

    pub fn from_vec1(v: Vec<f64>) -> Self {
        VarData {
            data: Array1::from_vec(v).into_dyn(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn ndim(&self) -> Dim<IxDynImpl> {
        self.data.dim()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Variable {
        self.to_node().reshape(shape)
    }

    pub fn transpose(&self) -> Variable {
        self.to_node().transpose()
    }

    pub fn exp(&self) -> Variable {
        let arg = self.to_node();
        arg.exp()
    }

    pub fn powi(&self, factor: i32) -> Variable {
        let arg = self.to_node();
        arg.powi(factor)
    }

    pub fn to_node(&self) -> Variable {
        Variable::from(self.clone())
    }

    pub fn is_same(&self, other: &Self, epsilon: f64) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data.flatten();
        let b = other.data.flatten();
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() <= epsilon)
    }
}

impl Eq for VarData {}

impl PartialEq for VarData {
    fn eq(&self, other: &Self) -> bool {
        self.data.as_ptr() == other.data.as_ptr()
    }
}

impl Hash for VarData {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.as_ptr().hash(state);
    }
}

struct FuncPtr(*const ());

impl FuncPtr {
    fn new(rc: &Rc<dyn Function>) -> Self {
        FuncPtr(Rc::as_ptr(rc) as *const ())
    }
}

impl PartialEq for FuncPtr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for FuncPtr {}

impl Hash for FuncPtr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

struct FunctionQueue {
    queue: VecDeque<Rc<dyn Function>>,
    set: HashSet<FuncPtr>,
}

impl FunctionQueue {
    pub fn new() -> Self {
        FunctionQueue {
            queue: VecDeque::new(),
            set: HashSet::new(),
        }
    }

    pub fn contains(&self, f: Rc<dyn Function>) -> bool {
        let ptr = FuncPtr::new(&f);
        self.set.contains(&ptr)
    }

    pub fn pop_front(&mut self) -> Option<Rc<dyn Function>> {
        let f = self.queue.pop_front();
        if let Some(frc) = f.clone() {
            self.set.remove(&FuncPtr::new(&frc));
        }
        f
    }

    pub fn push_back(&mut self, f: Rc<dyn Function>) {
        self.set.insert(FuncPtr::new(&f));
        self.queue.push_back(f);
    }
}

pub struct Variable {
    pub content: Rc<RefCell<VarData>>,
    pub grad: Rc<RefCell<Option<Variable>>>,
    pub retain_grad: bool,
    pub create_graph: bool,
    pub creator: Option<Rc<dyn Function>>,
}

impl Variable {
    pub fn from(vardata: VarData) -> Self {
        Variable {
            content: Rc::new(RefCell::new(vardata)),
            grad: Rc::new(RefCell::new(None)),
            retain_grad: false,
            create_graph: false,
            creator: None,
        }
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        let base = Array::zeros(shape);
        Self::from(VarData::new(base.into_dyn()))
    }

    pub fn zero(shape: usize) -> Self {
        let base = Array::zeros(shape);
        Self::from(VarData::new(base.into_dyn()))
    }

    pub fn randn(shape: (usize, usize), uniform: (f64, f64)) -> Self {
        let base = Array::random(shape, Uniform::new(uniform.0, uniform.1));
        Self::from(VarData::new(base.into_dyn()))
    }

    pub fn from_arry(data: Array<f64, IxDyn>) -> Self {
        Self::from(VarData::new(data))
    }

    pub fn set_creator(&mut self, creator: Rc<dyn Function>) {
        self.creator = Some(creator);
    }

    pub fn backward(&mut self) {
        if self.grad.borrow().is_none() {
            let var = Variable::from_arry(Array::ones(self.data().shape()));
            *self.grad.borrow_mut() = Some(var);
        }
        let mut functions = FunctionQueue::new();
        if let Some(creator) = self.creator.as_mut() {
            functions.push_back(creator.clone());
        }
        let self_data_ptr = self.content.as_ptr();
        while let Some(f) = functions.pop_front() {
            let mut gys = Vec::new();
            for output in f.get_outputs().iter_mut() {
                let output_ptr = output.content.as_ptr();
                if ptr::eq(self_data_ptr, output_ptr) {
                    let gy = self.grad.borrow().clone().unwrap().clone();
                    gys.push(gy);
                } else {
                    let v_ref = output.clone();
                    let grad = v_ref.grad.clone();
                    if self.retain_grad {
                        *v_ref.grad.borrow_mut() = None;
                    }
                    if grad.borrow().is_none() {
                        let var = Variable::from_arry(Array::ones(v_ref.data().shape()));
                        gys.push(var);
                    } else {
                        let gy = grad.borrow().clone().unwrap().clone();
                        gys.push(gy);
                    };
                }
            }
            enable_backprop!(self.create_graph, {
                let gxs = f.backward(gys);
                let mut gxs = VecDeque::from(gxs);
                let mut vars = f.get_inputs();
                for var in vars.iter_mut() {
                    if let Some(gx) = gxs.pop_front() {
                        if var.grad.borrow().is_none() {
                            *var.grad.borrow_mut() = Some(gx)
                        } else {
                            let tmp = var.grad.borrow().clone().unwrap().clone();
                            let tmp = if ptr::eq(self_data_ptr, tmp.content.as_ptr()) {
                                self.clone()
                            } else {
                                tmp
                            };
                            *var.grad.borrow_mut() = Some(tmp + gx);
                        }
                        if let Some(creator) = var.creator.as_ref() {
                            if !functions.contains(creator.clone()) {
                                functions.push_back(creator.clone());
                            }
                        }
                    }
                }
            });
        }
    }

    pub fn cleargrad(&mut self) {
        *self.grad.borrow_mut() = None
    }

    pub fn enable_graph(&mut self) {
        self.create_graph = true;
    }

    pub fn disable_graph(&mut self) {
        self.create_graph = false;
    }

    pub fn grad(&self) -> Option<Variable> {
        if let Some(grad) = self.grad.borrow().clone() {
            Some(grad.clone())
        } else {
            None
        }
    }

    pub fn data(&self) -> Array<f64, IxDyn> {
        self.content.borrow().data.clone()
    }

    pub fn is_same(&self, other: &Self, epsilon: f64) -> bool {
        let v1 = self.content.borrow();
        let v2 = other.content.borrow();
        v1.is_same(&v2, epsilon)
    }

    pub fn set_data(&self, data: Array<f64, IxDyn>) {
        self.content.borrow_mut().data = data
    }

    pub fn get_grad_vec(&self) -> Vec<f64> {
        if let Some(grad) = self.grad.borrow().clone() {
            grad.data().into_raw_vec_and_offset().0
        } else {
            vec![]
        }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Variable {
        function::Reshape::new(shape).apply(self.clone())
    }

    pub fn transpose(&self) -> Variable {
        function::Transpose::new().apply(self.clone())
    }

    pub fn exp(&self) -> Variable {
        let mut operator = function::Exp::new();
        operator.apply(self.clone())
    }

    pub fn powi(&self, factor: i32) -> Variable {
        let mut operator = function::Pow::new(factor);
        operator.apply(self.clone())
    }
}

impl Eq for Variable {}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        self.content.as_ptr() == other.content.as_ptr()
    }
}

impl Hash for Variable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.content.as_ptr().hash(state);
    }
}

impl Add for Variable {
    type Output = Variable;
    fn add(self, rhs: Self) -> Self {
        let mut operator = function::Add::new();
        operator.apply(self, rhs)
    }
}

impl Add<f64> for Variable {
    type Output = Variable;
    fn add(self, rhs: f64) -> Self::Output {
        let dim = self.data().raw_dim();
        let rhs = VarData::new(Array::from_elem(dim, rhs).into_dyn()).to_node();
        self + rhs
    }
}

impl Add<Variable> for f64 {
    type Output = Variable;
    fn add(self, rhs: Variable) -> Self::Output {
        let dim = rhs.data().raw_dim();
        let lhs = VarData::new(Array::from_elem(dim, self).into_dyn()).to_node();
        rhs + lhs
    }
}

impl Mul for Variable {
    type Output = Variable;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut operator = function::Mul::new();
        operator.apply(self, rhs)
    }
}

impl Mul<f64> for Variable {
    type Output = Variable;
    fn mul(self, rhs: f64) -> Self::Output {
        let dim = self.data().raw_dim();
        let rhs = VarData::new(Array::from_elem(dim, rhs).into_dyn()).to_node();
        self * rhs
    }
}

impl Mul<Variable> for f64 {
    type Output = Variable;
    fn mul(self, rhs: Variable) -> Self::Output {
        let dim = rhs.data().raw_dim();
        let lhs = VarData::new(Array::from_elem(dim, self).into_dyn()).to_node();
        rhs * lhs
    }
}

impl Sub for Variable {
    type Output = Variable;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut operator = function::Sub::new();
        operator.apply(self, rhs)
    }
}

impl Sub<f64> for Variable {
    type Output = Variable;
    fn sub(self, rhs: f64) -> Self::Output {
        let dim = self.data().raw_dim();
        let rhs = VarData::new(Array::from_elem(dim, rhs).into_dyn()).to_node();
        self - rhs
    }
}

impl Sub<Variable> for f64 {
    type Output = Variable;
    fn sub(self, rhs: Variable) -> Self::Output {
        let dim = rhs.data().raw_dim();
        let lhs = VarData::new(Array::from_elem(dim, self).into_dyn()).to_node();
        rhs - lhs
    }
}

impl Div for Variable {
    type Output = Variable;
    fn div(self, rhs: Self) -> Self::Output {
        let mut operator = function::Div::new();
        operator.apply(self, rhs)
    }
}

impl Div<f64> for Variable {
    type Output = Variable;
    fn div(self, rhs: f64) -> Self::Output {
        let dim = self.data().raw_dim();
        let rhs = VarData::new(Array::from_elem(dim, rhs).into_dyn()).to_node();
        self / rhs
    }
}

impl Div<Variable> for f64 {
    type Output = Variable;
    fn div(self, rhs: Variable) -> Self::Output {
        let dim = rhs.data().raw_dim();
        let lhs = VarData::new(Array::from_elem(dim, self).into_dyn()).to_node();
        rhs / lhs
    }
}

impl Neg for Variable {
    type Output = Variable;
    fn neg(self) -> Self::Output {
        let mut operator = function::Neg::new();
        operator.apply(self)
    }
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        Variable {
            content: self.content.clone(),
            grad: self.grad.clone(),
            retain_grad: self.retain_grad,
            create_graph: self.create_graph,
            creator: self.creator.clone(),
        }
    }
}

impl Clone for VarData {
    fn clone(&self) -> Self {
        return VarData {
            data: self.data.clone(),
        };
    }
}

impl fmt::Display for VarData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Variable({})", self.data)
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = &self.data();
        write!(f, "Variable({})", data)
    }
}

impl Add for VarData {
    type Output = Variable;
    fn add(self, rhs: Self) -> Self::Output {
        let x1 = self.to_node();
        let x2 = rhs.to_node();
        x1.add(x2)
    }
}

impl Mul for VarData {
    type Output = Variable;
    fn mul(self, rhs: Self) -> Self::Output {
        let x1 = self.to_node();
        let x2 = rhs.to_node();
        x1.mul(x2)
    }
}

impl Div for VarData {
    type Output = Variable;
    fn div(self, rhs: Self) -> Self::Output {
        let x1 = self.to_node();
        let x2 = rhs.to_node();
        x1.div(x2)
    }
}

impl Sub for VarData {
    type Output = Variable;
    fn sub(self, rhs: Self) -> Self::Output {
        let x1 = self.to_node();
        let x2 = rhs.to_node();
        x1.sub(x2)
    }
}

impl Neg for VarData {
    type Output = Variable;
    fn neg(self) -> Self::Output {
        self.to_node().neg()
    }
}
