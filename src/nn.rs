use std::{collections::HashSet, usize};

use ndarray::Array;

use crate::core::variable::{VarNode, Variable};

pub trait Layer {
    fn forward(&self, x: VarNode) -> VarNode;
}

pub trait Learnable {
    fn parameters(&self) -> HashSet<Variable>;
}

struct Linear {
    input: usize,
    output: usize,
    w: VarNode,
    b: VarNode,
}

impl Linear {
    pub fn new(input: usize, output: usize) -> Self {
        let w_base = Array::zeros((input, output)).into_dyn();
        let w = Variable::new(w_base).to_node();
        let b_base = Array::zeros((output, 1)).into_dyn();
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
        w_node * x + b_node
    }
}

impl Learnable for Linear {
    fn parameters(&self) -> HashSet<Variable> {
        let mut set = HashSet::new();
        let w = self.w.content.borrow().clone();
        set.insert(w);
        let b = self.w.content.borrow().clone();
        set.insert(b);
        set
    }
}
