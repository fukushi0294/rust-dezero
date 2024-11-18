use std::collections::{HashMap, HashSet};

use crate::{
    core::variable::{VarNode, VarData},
    enable_backprop, utils::WeakKey,
};

pub trait Optimizer {
    fn zero_grad(&self);
    fn step(&mut self);
}

pub struct SGD {
    lr: f64,
    params: HashSet<VarNode>,
}

impl SGD {
    pub fn new(lr: f64, params: HashSet<VarNode>) -> Self {
        SGD { lr, params }
    }
}

impl Optimizer for SGD {
    fn zero_grad(&self) {
        for var in self.params.iter() {
            var.cleargrad();
        }
    }

    fn step(&mut self) {
        for var in self.params.iter() {
            let grad = var.grad().unwrap().data();
            let data = var.data();
            var.set_data(data - self.lr * grad);
        }
    }
}

pub struct MomentumSGD {
    lr: f64,
    delta: f64,
    vs: HashMap<WeakKey<VarData>, VarData>,
    params: HashSet<VarNode>,
}

impl MomentumSGD {
    pub fn new(lr: f64, delta: f64, params: HashSet<VarNode>) -> Self {
        MomentumSGD {
            lr,
            delta,
            vs: HashMap::new(),
            params,
        }
    }
}

impl Optimizer for MomentumSGD {
    fn zero_grad(&self) {
        for var in self.params.iter() {
            var.enable_graph();
            var.cleargrad();
        }
    }

    fn step(&mut self) {
        for var in self.params.iter() {
            enable_backprop!(false, {
                let grad = var.grad().unwrap();
                let grad_ref = &grad.content;
                let data = grad_ref.borrow().clone();
                let grad_key = WeakKey::from(&grad_ref);

                let v = if let Some(v_old) = self.vs.get(&grad_key) {
                    self.delta * v_old.clone().to_node() - self.lr * grad
                } else {
                    -self.lr * grad
                };

                self.vs.insert(grad_key, data);
                let w = var.clone() + v;
                var.set_data(w.data());
            });
        }
    }
}
