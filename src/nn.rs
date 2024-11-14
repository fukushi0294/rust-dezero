use std::{collections::HashSet, usize};

use crate::core::function as F;
use crate::core::variable::{VarNode, Variable};
use ndarray::Array;

pub trait Layer {
    fn forward(&self, x: VarNode) -> VarNode;
    fn parameters(&self) -> HashSet<Variable> {
        HashSet::new()
    }
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

    fn parameters(&self) -> HashSet<Variable> {
        let mut set = HashSet::new();
        let w = self.w.content.borrow().clone();
        set.insert(w);
        let b = self.w.content.borrow().clone();
        set.insert(b);
        set
    }
}

struct Sigmoid {}

impl Layer for Sigmoid {
    fn forward(&self, x: VarNode) -> VarNode {
        return 1.0 / (1.0 + F::exp(-x));
    }
}
