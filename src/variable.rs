use crate::function::{self, Function};
use ndarray::{Array, IxDyn};
use ndarray::{Dim, IxDynImpl};
use std::cell::{RefCell, RefMut};
use std::collections::VecDeque;
use std::fmt;
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

    pub fn ndim(&self) -> Dim<IxDynImpl> {
        self.data.dim()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn set_creator(&mut self, creator: Rc<dyn Function>) {
        self.creator = Some(creator);
    }

    pub fn cleargrad(&mut self) {
        self.grad = None
    }

    pub fn to_placeholder(&self) -> PlaceHolder {
        PlaceHolder {
            content: vec![Rc::new(RefCell::new(self.clone()))],
        }
    }

    pub fn backward(&mut self) -> Option<Rc<RefCell<Variable>>> {
        if self.grad.is_none() {
            self.grad = Some(Array::ones(self.data.shape()))
        }
        let mut functions = VecDeque::new();
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
                    if let Some(creator) = vref.creator.as_mut() {
                        functions.push_back(creator.clone());
                    } else {
                        return Some(Rc::clone(var));
                    }
                }
            }
        }
        None
    }
}

pub struct PlaceHolder {
    pub content: Vec<Rc<RefCell<Variable>>>,
}

impl PlaceHolder {
    pub fn len(&self) -> usize {
        self.content.len()
    }

    pub fn concat(&self, other: PlaceHolder) -> PlaceHolder {
        let mut content = self.clone().content;
        for c in other.content.iter() {
            content.push(c.clone());
        }
        PlaceHolder { content }
    }

    pub fn get(&self, index: usize) -> RefMut<Variable> {
        let content = self.content.get(index);
        assert!(&content.is_some(), "content does'nt exist.");
        content.unwrap().borrow_mut()
    }
}

impl Add for PlaceHolder {
    type Output = PlaceHolder;
    fn add(self, rhs: Self) -> Self {
        let args = self.concat(rhs);
        let mut operator = function::Add::new();
        operator.apply(args)
    }
}

impl Mul for PlaceHolder {
    type Output = PlaceHolder;
    fn mul(self, rhs: Self) -> Self::Output {
        let args = self.concat(rhs);
        let mut operator = function::Mul::new();
        operator.apply(args)
    }
}

impl Sub for PlaceHolder {
    type Output = PlaceHolder;
    fn sub(self, rhs: Self) -> Self::Output {
        let args = self.concat(rhs);
        let mut operator = function::Sub::new();
        operator.apply(args)
    }
}

impl Div for PlaceHolder {
    type Output = PlaceHolder;
    fn div(self, rhs: Self) -> Self::Output {
        let args = self.concat(rhs);
        let mut operator = function::Div::new();
        operator.apply(args)
    }
}

impl Neg for PlaceHolder {
    type Output = PlaceHolder;
    fn neg(self) -> Self::Output {
        let mut operator = function::Sub::new();
        operator.apply(self)
    }
}

impl Clone for PlaceHolder {
    fn clone(&self) -> Self {
        let mut output = Vec::new();
        for c in self.content.iter() {
            output.push(c.clone());
        }
        PlaceHolder { content: output }
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
    type Output = PlaceHolder;
    fn add(self, rhs: Self) -> Self::Output {
        let x1 = self.to_placeholder();
        let x2 = rhs.to_placeholder();
        x1.add(x2)
    }
}

impl Mul for Variable {
    type Output = PlaceHolder;
    fn mul(self, rhs: Self) -> Self::Output {
        let x1 = self.to_placeholder();
        let x2 = rhs.to_placeholder();
        x1.mul(x2)
    }
}

impl Div for Variable {
    type Output = PlaceHolder;
    fn div(self, rhs: Self) -> Self::Output {
        let x1 = self.to_placeholder();
        let x2 = rhs.to_placeholder();
        x1.div(x2)
    }
}

impl Sub for Variable {
    type Output = PlaceHolder;
    fn sub(self, rhs: Self) -> Self::Output {
        let x1 = self.to_placeholder();
        let x2 = rhs.to_placeholder();
        x1.sub(x2)
    }
}

impl Neg for Variable {
    type Output = PlaceHolder;
    fn neg(self) -> Self::Output {
        self.to_placeholder().neg()
    }
}
