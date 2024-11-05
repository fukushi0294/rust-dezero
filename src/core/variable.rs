use crate::core::function::{self, BiFunction, Function, UniFunction};
use ndarray::{Array, Array1, IxDyn};
use ndarray::{Dim, IxDynImpl};
use std::cell::{RefCell, RefMut};
use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::ptr;
use std::rc::Rc;

pub struct Variable {
    pub data: Array<f64, IxDyn>,
    pub grad: Option<VarNode>,
    pub retain_grad: bool,
    pub creator: Option<Rc<dyn Function>>,
}

impl Variable {
    pub fn new(data: Array<f64, IxDyn>) -> Self {
        return Variable {
            data: data,
            grad: None,
            retain_grad: false,
            creator: None,
        };
    }

    pub fn from_vec1(v: Vec<f64>) -> Self {
        Variable {
            data: Array1::from_vec(v).into_dyn(),
            grad: None,
            retain_grad: false,
            creator: None,
        }
    }

    pub fn ndim(&self) -> Dim<IxDynImpl> {
        self.data.dim()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn exp(&self) -> VarNode {
        let arg = self.to_node();
        arg.exp()
    }

    pub fn powi(&self, factor: i32) -> VarNode {
        let arg = self.to_node();
        arg.powi(factor)
    }

    pub fn set_creator(&mut self, creator: Rc<dyn Function>) {
        self.creator = Some(creator);
    }

    pub fn cleargrad(&mut self) {
        self.grad = None
    }

    pub fn to_node(&self) -> VarNode {
        VarNode {
            content: Rc::new(RefCell::new(self.clone())),
        }
    }

    pub fn backward(&mut self) {
        if self.grad.is_none() {
            let var = Variable::new(Array::ones(self.data.shape()));
            self.grad = Some(var.to_node());
        }
        let mut functions = FunctionQueue::new();
        if let Some(creator) = self.creator.as_mut() {
            functions.push_back(creator.clone());
        }
        let self_ptr = self as *mut Variable;
        while let Some(f) = functions.pop_front() {
            let mut gys = Vec::new();
            for output in f.get_outputs().iter_mut() {
                let output_ptr = output.as_ptr();
                if ptr::eq(self_ptr, output_ptr) {
                    gys.push(self.grad.clone().unwrap().content.borrow().data.clone());
                } else {
                    let mut v_ref = output.borrow_mut();
                    let grad = v_ref.grad.clone();
                    if self.retain_grad {
                        v_ref.grad = None
                    }
                    if grad.is_none() {
                        gys.push(Array::ones(v_ref.data.shape()));
                    } else {
                        gys.push(grad.unwrap().content.borrow().data.clone());
                    };
                }
            }
            let gxs = f.backward(&gys);
            let mut gxs = VecDeque::from(gxs);
            let mut vars = f.get_inputs();
            for var in vars.iter_mut() {
                if let Some(gx) = gxs.pop_front() {
                    let mut vref = var.borrow_mut();
                    let gx = Variable::new(gx).to_node();
                    if vref.grad.is_none() {
                        vref.grad = Some(gx)
                    } else {
                        let tmp = vref.grad.clone().unwrap();
                        vref.grad = Some(tmp + gx);
                    }
                    if let Some(creator) = vref.creator.as_ref() {
                        if !functions.contains(creator.clone()) {
                            functions.push_back(creator.clone());
                        }
                    }
                }
            }
        }
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

pub struct VarNode {
    pub content: Rc<RefCell<Variable>>,
}

impl VarNode {
    pub fn extract(&self) -> RefMut<Variable> {
        self.content.borrow_mut()
    }

    pub fn get_grad_vec(&self) -> Vec<f64> {
        let var = self.extract();
        if let Some(grad) = &var.grad {
            grad.content
                .borrow()
                .data
                .clone()
                .into_raw_vec_and_offset()
                .0
        } else {
            vec![]
        }
    }

    pub fn exp(&self) -> VarNode {
        let mut operator = function::Exp::new();
        operator.apply(self.clone())
    }

    pub fn powi(&self, factor: i32) -> VarNode {
        let mut operator = function::Pow::new(factor);
        operator.apply(self.clone())
    }

    pub fn blanch(&self) -> (VarNode, VarNode) {
        let mut operator = function::Blanch::new();
        operator.apply(self.clone())
    }
}

impl Add for VarNode {
    type Output = VarNode;
    fn add(self, rhs: Self) -> Self {
        let mut operator = function::Add::new();
        operator.apply(self, rhs)
    }
}

impl Mul for VarNode {
    type Output = VarNode;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut operator = function::Mul::new();
        operator.apply(self, rhs)
    }
}

impl Mul<f64> for VarNode {
    type Output = VarNode;
    fn mul(self, rhs: f64) -> Self::Output {
        let dim = self.content.borrow().data.raw_dim();
        let rhs = Variable::new(Array::from_elem(dim, rhs).into_dyn()).to_node();
        self * rhs
    }
}

impl Mul<VarNode> for f64 {
    type Output = VarNode;
    fn mul(self, rhs: VarNode) -> Self::Output {
        let dim = rhs.content.borrow().data.raw_dim();
        let lhs = Variable::new(Array::from_elem(dim, self).into_dyn()).to_node();
        rhs * lhs
    }
}

impl Sub for VarNode {
    type Output = VarNode;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut operator = function::Sub::new();
        operator.apply(self, rhs)
    }
}

impl Div for VarNode {
    type Output = VarNode;
    fn div(self, rhs: Self) -> Self::Output {
        let mut operator = function::Div::new();
        operator.apply(self, rhs)
    }
}

impl Neg for VarNode {
    type Output = VarNode;
    fn neg(self) -> Self::Output {
        let mut operator = function::Neg::new();
        operator.apply(self)
    }
}

impl Clone for VarNode {
    fn clone(&self) -> Self {
        VarNode {
            content: self.content.clone(),
        }
    }
}

impl Clone for Variable {
    fn clone(&self) -> Self {
        return Variable {
            data: self.data.clone(),
            grad: self.grad.clone(),
            retain_grad: self.retain_grad,
            creator: None,
        };
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Variable({}", self.data)
    }
}

impl Add for Variable {
    type Output = VarNode;
    fn add(self, rhs: Self) -> Self::Output {
        let x1 = self.to_node();
        let x2 = rhs.to_node();
        x1.add(x2)
    }
}

impl Mul for Variable {
    type Output = VarNode;
    fn mul(self, rhs: Self) -> Self::Output {
        let x1 = self.to_node();
        let x2 = rhs.to_node();
        x1.mul(x2)
    }
}

impl Div for Variable {
    type Output = VarNode;
    fn div(self, rhs: Self) -> Self::Output {
        let x1 = self.to_node();
        let x2 = rhs.to_node();
        x1.div(x2)
    }
}

impl Sub for Variable {
    type Output = VarNode;
    fn sub(self, rhs: Self) -> Self::Output {
        let x1 = self.to_node();
        let x2 = rhs.to_node();
        x1.sub(x2)
    }
}

impl Neg for Variable {
    type Output = VarNode;
    fn neg(self) -> Self::Output {
        self.to_node().neg()
    }
}
