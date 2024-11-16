use std::cell::RefCell;
use std::rc::Rc;
use std::{collections::HashSet, usize};

use crate::core::function::{self as F, Function, ParamSupplier, UniFunction};
use crate::core::variable::{VarNode, Variable};
use crate::params;
use derives::{Learnable, UniFunction};
use ndarray::{Array, IxDyn};

pub trait Layer {
    fn forward(&self, x: VarNode) -> VarNode;
}

pub trait Learnable {
    fn parameters(&self) -> HashSet<VarNode> {
        HashSet::new()
    }
}

#[derive(Learnable)]
pub struct Linear {
    input: usize,
    output: usize,
    w: VarNode,
    b: VarNode,
}

impl Linear {
    pub fn new(input: usize, output: usize) -> Self {
        let w = Variable::zeros((input, output)).to_node();
        let b = Variable::zero(output).to_node();
        Linear {
            input,
            output,
            w,
            b,
        }
    }
}

impl Layer for Linear {
    fn forward(&self, x: VarNode) -> VarNode {
        let w_node = self.w.clone();
        let b_node = self.b.clone();
        F::matmal(x, w_node) + b_node
    }
}

#[derive(UniFunction)]
pub struct Sigmoid {
    input: Option<Rc<RefCell<Variable>>>,
    output: Option<Rc<RefCell<Variable>>>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {
            input: None,
            output: None,
        }
    }
}

impl Function for Sigmoid {
    fn new_instance(
        &self,
        inputs: &[Rc<RefCell<Variable>>],
        outputs: &[Rc<RefCell<Variable>>],
    ) -> Rc<dyn Function> {
        let x = inputs[0].clone();
        let y = outputs[0].clone();
        let f = Sigmoid {
            input: Some(x),
            output: Some(y),
        };
        Rc::new(f)
    }
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        let y = 1.0 / (1.0 + (-x).exp());
        vec![y]
    }
    fn backward(&self, gys: Vec<VarNode>) -> Vec<VarNode> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        let y = VarNode {
            content: self.output.clone().unwrap().clone(),
        };
        return vec![gys[0].clone() * y.clone() * (1. - y)];
    }
    fn supplyer(&self) -> Rc<dyn ParamSupplier> {
        params!(
            (self.input.clone().unwrap()),
            (self.output.clone().unwrap())
        )
    }
}
