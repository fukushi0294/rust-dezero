use crate::function::Function;
use ndarray::{Array, IxDyn};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::f64::consts::E;
use std::ptr;
use std::rc::Rc;

pub struct Variable {
    pub data: Array<f64, IxDyn>,
    pub grad: Option<Array<f64, IxDyn>>,
    pub creator: Option<Rc<dyn Function>>,
}

impl Variable {
    pub fn new(data: Array<f64, IxDyn>) -> Self {
        return Variable {
            data: data,
            grad: None,
            creator: None,
        };
    }

    pub fn set_creator(&mut self, creator: Rc<dyn Function>) {
        self.creator = Some(creator);
    }

    pub fn cleargrad(&mut self) {
        self.grad = None
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
            for output in f.get_outputs().iter() {
                let output_ptr = output.as_ptr();
                if ptr::eq(self_ptr, output_ptr) {
                    gys.push(self.grad.clone().unwrap());
                } else {
                    let v_ref = output.borrow_mut();
                    let grad = v_ref.grad.clone();
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

impl Clone for Variable {
    fn clone(&self) -> Self {
        return Variable {
            data: self.data.clone(),
            grad: self.grad.clone(),
            creator: None,
        };
    }
}
