use std::cell::RefCell;
use std::rc::Rc;
use std::{collections::HashSet, usize};

use crate::core::function::{self as F, Function, FunctionNode, Sum, UniFunction};
use crate::core::variable::{VarData, Variable};
use derives::{FunctionNode, Learnable, UniFunction};
use ndarray::{Array, Axis, IxDyn};

pub trait Layer {
    fn forward(&self, x: Variable) -> Variable;
}

pub trait Learnable {
    fn parameters(&self) -> HashSet<Variable> {
        HashSet::new()
    }
}

#[derive(Learnable)]
pub struct Linear {
    input: usize,
    output: usize,
    w: Variable,
    b: Variable,
}

impl Linear {
    pub fn new(input: usize, output: usize) -> Self {
        let w = Variable::zeros((input, output));
        let b = Variable::zero(output);
        Linear {
            input,
            output,
            w,
            b,
        }
    }
}

impl Layer for Linear {
    fn forward(&self, x: Variable) -> Variable {
        let w_node = self.w.clone();
        let b_node = self.b.clone();
        F::matmal(x, w_node) + b_node
    }
}

pub trait Module: Layer + Learnable {}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

impl Layer for Sequential {
    fn forward(&self, x: Variable) -> Variable {
        let mut x_out = x.clone();
        for l in self.layers.iter() {
            x_out = l.forward(x_out.clone());
        }
        x_out
    }
}

impl Learnable for Sequential {
    fn parameters(&self) -> HashSet<Variable> {
        self.layers
            .iter()
            .flat_map(|l| l.parameters().into_iter())
            .map(|e| e.clone())
            .collect()
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct Sigmoid {
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
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
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        let y = 1.0 / (1.0 + (-x).exp());
        vec![y]
    }
    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        let y = self.output.clone().unwrap().clone();
        return vec![gys[0].clone() * y.clone() * (1. - y)];
    }
}

#[derive(UniFunction, FunctionNode)]
pub struct SoftMax {
    #[node_I]
    input: Option<Variable>,
    #[node_O]
    output: Option<Variable>,
    axis: usize,
}

impl SoftMax {
    pub fn new(axis: usize) -> Self {
        SoftMax {
            input: None,
            output: None,
            axis,
        }
    }
}

impl Function for SoftMax {
    fn forward(&self, inputs: &[Array<f64, IxDyn>]) -> Vec<Array<f64, IxDyn>> {
        assert!(inputs.len() == 1, "inputs slice size must be 1");
        let x = inputs[0].clone();
        let y = x.view().exp();
        let y_sum = y.clone().to_owned().sum_axis(Axis(self.axis));
        vec![y / y_sum]
    }

    fn backward(&self, gys: Vec<Variable>) -> Vec<Variable> {
        assert!(gys.len() == 1, "inputs slice size must be 1");
        let y = self.output.clone().unwrap().clone();
        let gy = gys[0].clone();
        let gx = y.clone() * gy;
        let mut sum = Sum::new_axis_keep_dim(self.axis, true);
        let sumdx = sum(gx.clone());
        let gx = gx - y * sumdx;
        return vec![gx];
    }
}
