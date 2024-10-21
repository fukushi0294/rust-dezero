use crate::variable::Variable;
use ndarray::{Array, IxDyn};

pub trait Function {
    fn call(&mut self, input: &Variable) -> Variable {
        let x = input.data.clone();
        let y = self.forward(&x);
        let mut output = Variable::new(y);
        output.set_creator(self.new_instance(input));
        output
    }
    fn new_instance(&self, input: &Variable) -> Box<dyn Function>;
    fn forward(&self, input: &Array<f64,IxDyn>) -> Array<f64, IxDyn>;
    fn backward(&mut self, gy: &Array<f64,IxDyn>) -> Array<f64, IxDyn>;
    fn get_input(&mut self) -> Option<&mut Variable>;
}

#[derive(Default)]
pub struct Square {
    input: Option<Variable>,
}

impl Function for Square {
    fn new_instance(&self, input: &Variable) -> Box<dyn Function> {
        Box::new(Square {
            input: Some(Variable::new(input.data.clone())),
        })
    }
    fn forward(&self, input: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        input.pow2()
    }
    fn backward(&mut self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let x = self.input.as_ref().unwrap();
        2.0 * x.data.clone() * gy
    }
    fn get_input(&mut self) -> Option<&mut Variable> {
        self.input.as_mut()
    }
}


pub fn square(x: &Variable)-> Variable {
    let mut f = Square::default();
    f.call(x)
}

#[derive(Default)]
pub struct Exp {
    input: Option<Variable>,
}

impl Function for Exp {
    fn new_instance(&self, input: &Variable) -> Box<dyn Function> {
        Box::new(Exp {
            input: Some(Variable::new(input.data.clone())),
        })
    }
    fn forward(&self, input: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        input.exp()
    }
    fn backward(&mut self, gy: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
        let x = self.input.as_ref().unwrap();
        x.data.exp() * gy
    }
    fn get_input(&mut self) -> Option<&mut Variable> {
        self.input.as_mut()
    }
}

pub fn exp(x: &Variable) -> Variable {
    let mut f = Exp::default();
    f.call(x)
}

fn numerical_diff(f: &mut impl Function, x: Variable) -> Array<f64, IxDyn> {
    let eps = 1e-4;
    let x0 = Variable::new(x.data.clone() - eps);
    let x1 = Variable::new(x.data.clone() + eps);
    let y0 = f.call(&x0);
    let y1 = f.call(&x1);
    return (y1.data - y0.data) / (eps * 2.0);
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    #[test]
    fn square_test() {
        let x1 = vec![5.0, 10.0];
        let expected: Vec<f64> = x1.iter().map(|&x| x * x).collect();
        let x2 = Variable::new(Array1::from_vec(x1).into_dyn());
        let mut f = Square::default();
        let actual = f.call(&x2);
        let s = actual.data.as_slice();
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn backward_test() {
        let x = Array1::from_vec(vec![10.0]);
        let xv = x.clone().to_vec();
        let expected: Vec<f64> = x.to_vec().iter().map(|x| 2.0*x).collect();
        let mut f = Square::default();
        let mut y = f.call(&Variable::new(x.into_dyn()));
        let result = y.backward();
        assert_eq!(true, result.is_some());
        let data = result.unwrap().data.as_slice();
        assert_eq!(xv, data.unwrap());
        let actual = result.unwrap().grad.as_slice();
        assert_eq!(expected, actual.unwrap())
    }
}
