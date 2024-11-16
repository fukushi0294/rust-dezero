use std::collections::HashSet;

use crate::core::variable::VarNode;

pub trait Optimizer {
    fn zero_grad(&self);
    fn step(&self);
}

pub struct SGD {
    lr: f64,
    params: HashSet<VarNode>,
}

impl SGD {
    pub fn new(lr:f64, params:HashSet<VarNode>) -> Self {
        SGD {
            lr,
            params
        }
    }
}

impl Optimizer for SGD {
    fn zero_grad(&self) {
        for var in self.params.iter() {
            var.cleargrad();
        }
    }

    fn step(&self) {
        for var in self.params.iter() {
            let grad = var.grad().unwrap().data();
            let data = var.data();
            var.set_data(data - self.lr * grad);
        }
    }
}
