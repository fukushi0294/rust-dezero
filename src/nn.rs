use std::{collections::HashSet, usize};

use crate::core::function::{self as F};
use crate::core::variable::Variable;
use derives::{Learnable, Module};

pub trait Layer {
    fn forward(&self, x: Variable) -> Variable;
}

pub trait Learnable {
    fn parameters(&self) -> HashSet<Variable> {
        HashSet::new()
    }
}

pub trait Module: Layer + Learnable {}

#[derive(Learnable, Module)]
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

#[derive(Module)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

#[macro_export]
macro_rules! sequential {
    ($($layer:expr),* $(,)?) => {
        {
            let layers: Vec<Box<dyn $crate::nn::Module>> = vec![
                $(Box::new($layer)),*
            ];
            $crate::nn::Sequential::new(layers)
        }
    };
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

#[derive(Learnable, Module)]
pub struct Sigmoid {}

impl Layer for Sigmoid {
    fn forward(&self, x: Variable) -> Variable {
        F::Sigmoid::new()(x)
    }
}

#[derive(Learnable, Module)]
pub struct Softmax {
    axis: usize,
}

impl Softmax {
    fn new(axis: usize) -> Self {
        Softmax { axis }
    }
}

impl Layer for Softmax {
    fn forward(&self, x: Variable) -> Variable {
        F::Softmax::new(self.axis)(x)
    }
}
