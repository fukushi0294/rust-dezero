use crate::core::function::{self, Function};
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
    pub grad: Option<Array<f64, IxDyn>>,
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
            content: vec![Rc::new(RefCell::new(self.clone()))],
        }
    }

    pub fn backward(&mut self) {
        if self.grad.is_none() {
            self.grad = Some(Array::ones(self.data.shape()))
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
                    gys.push(self.grad.clone().unwrap());
                } else {
                    let mut v_ref = output.borrow_mut();
                    let grad = v_ref.grad.clone();
                    if self.retain_grad {
                        v_ref.grad = None
                    }
                    let grad = if grad.is_none() {
                        Array::ones(v_ref.data.shape())
                    } else {
                        grad.clone().unwrap()
                    };
                    gys.push(grad.clone());
                }
            }
            let gxs = f.backward(&gys);
            let mut gxs = VecDeque::from(gxs);
            let mut vars = f.get_inputs();
            for var in vars.iter_mut() {
                if let Some(gx) = gxs.pop_front() {
                    let mut vref = var.borrow_mut();
                    if vref.grad.is_none() {
                        vref.grad = Some(gx)
                    } else {
                        let tmp = &vref.grad.clone().unwrap();
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
    pub content: Vec<Rc<RefCell<Variable>>>,
}

impl VarNode {
    pub fn len(&self) -> usize {
        self.content.len()
    }

    pub fn concat(&self, other: VarNode) -> VarNode {
        let mut content = self.clone().content;
        for c in other.content.iter() {
            content.push(c.clone());
        }
        VarNode { content }
    }

    pub fn get(&self, index: usize) -> RefMut<Variable> {
        let content = self.content.get(index);
        assert!(&content.is_some(), "content does'nt exist.");
        content.unwrap().borrow_mut()
    }

    pub fn get_grad_as_vec(&self, index: usize) -> Vec<f64> {
        let var = self.get(index);
        if let Some(grad) = &var.grad {
            grad.clone().into_raw_vec_and_offset().0
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

    pub fn duplicate(&self) -> VarNode {
        let mut operator = function::Blanch::new();
        assert!(self.content.len() == 1);
        operator.apply(self.clone())
    }

    pub fn blanch(&self) -> (VarNode, VarNode) {
        let node = self.duplicate();
        let content1 = node.content.get(0).unwrap().clone();
        let content2 = node.content.get(1).unwrap().clone();
        (
            VarNode {
                content: vec![content1],
            },
            VarNode {
                content: vec![content2],
            },
        )
    }
}

impl Add for VarNode {
    type Output = VarNode;
    fn add(self, rhs: Self) -> Self {
        let args = self.concat(rhs);
        let mut operator = function::Add::new();
        operator.apply(args)
    }
}

impl Mul<f64> for VarNode {
    type Output = VarNode;
    fn mul(self, rhs: f64) -> Self::Output {
        let vars = self.content;
        let mut output = Vec::new();
        for var in vars.iter() {
            var.borrow_mut().data *= rhs;
            output.push(var.clone());
        }
        VarNode { content: output }
    }
}

impl Mul<VarNode> for f64 {
    type Output = VarNode;
    fn mul(self, rhs: VarNode) -> Self::Output {
        let vars = rhs.content;
        let mut output = Vec::new();
        for var in vars.iter() {
            var.borrow_mut().data *= self;
            output.push(var.clone());
        }
        VarNode { content: output }
    }
}

impl Mul for VarNode {
    type Output = VarNode;
    fn mul(self, rhs: Self) -> Self::Output {
        let args = self.concat(rhs);
        let mut operator = function::Mul::new();
        operator.apply(args)
    }
}

impl Sub for VarNode {
    type Output = VarNode;
    fn sub(self, rhs: Self) -> Self::Output {
        let args = self.concat(rhs);
        let mut operator = function::Sub::new();
        operator.apply(args)
    }
}

impl Div for VarNode {
    type Output = VarNode;
    fn div(self, rhs: Self) -> Self::Output {
        let args = self.concat(rhs);
        let mut operator = function::Div::new();
        operator.apply(args)
    }
}

impl Neg for VarNode {
    type Output = VarNode;
    fn neg(self) -> Self::Output {
        let mut operator = function::Sub::new();
        operator.apply(self)
    }
}

impl Clone for VarNode {
    fn clone(&self) -> Self {
        let mut output = Vec::new();
        for c in self.content.iter() {
            output.push(c.clone());
        }
        VarNode { content: output }
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
