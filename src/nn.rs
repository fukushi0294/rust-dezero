extern crate proc_macro;
use std::cell::RefCell;
use std::rc::Rc;
use std::{collections::HashSet, usize};

use crate::core::function as F;
use crate::core::variable::{VarNode, Variable};
use ndarray::Array;

pub trait Layer {
    fn forward(&self, x: VarNode) -> VarNode;
    fn parameters(&self) -> HashSet<VarNode> {
        HashSet::new()
    }
}
pub struct Linear {
    input: usize,
    output: usize,
    w: VarNode,
    b: VarNode,
}

impl Linear {
    pub fn new(input: usize, output: usize) -> Self {
        let w_base = Array::zeros((input, output)).into_dyn();
        let w = Variable::new(w_base).to_node();
        let b_base = Array::zeros(output).into_dyn();
        let b = Variable::new(b_base).to_node();
        Linear {
            input,
            output,
            w: w,
            b: b,
        }
    }
}

impl Layer for Linear {
    fn forward(&self, x: VarNode) -> VarNode {
        let w_node = self.w.clone();
        let b_node = self.b.clone();
        F::matmal(x, w_node) + b_node
    }
    fn parameters(&self) -> HashSet<VarNode> {
        let mut set = HashSet::new();
        set.insert(self.w.clone());
        set.insert(self.b.clone());
        set
    }
}

pub struct Sigmoid {}

impl Layer for Sigmoid {
    fn forward(&self, x: VarNode) -> VarNode {
        return 1.0 / (1.0 + F::exp(-x));
    }
}
