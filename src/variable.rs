use std::collections::VecDeque;

use crate::function::Function;

pub struct Variable {
    pub data: f64,
    pub grad: f64,
    pub creator: Option<Box<dyn Function>>,
}

impl Variable {
    pub fn new(data: f64) -> Self {
        return Variable {
            data: data,
            grad: 1.0,
            creator: None,
        };
    }

    pub fn set_creator(&mut self, creator: Box<dyn Function>) {
        self.creator = Some(creator);
    }

    pub fn backward(&mut self) -> Option<&Variable> {
        let mut functions = VecDeque::new();
        functions.push_back(self.creator.as_mut()?);
        let grad = self.grad;
        while let Some(f) = functions.pop_front() {
            let grad = f.backward(grad);
            let var = f.get_input()?;
            var.grad = grad;
            if var.creator.is_none() {
                return Some(var);
            }
            functions.push_back(var.creator.as_mut()?);
        }
        None
    }
}
