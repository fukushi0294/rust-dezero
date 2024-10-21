use crate::variable::Variable;

pub trait Function {
    fn call(&mut self, input: &Variable) -> Variable {
        let x = input.data;
        let y = self.forward(x);
        let mut output = Variable::new(y);
        output.set_creator(self.new_instance(input));
        output
    }
    fn new_instance(&self, input: &Variable) -> Box<dyn Function>;
    fn forward(&self, input: f64) -> f64;
    fn backward(&mut self, gy: f64) -> f64;
    fn get_input(&mut self) -> Option<&mut Variable>;
}

#[derive(Default)]
pub struct Square {
    input: Option<Variable>,
}

impl Function for Square {
    fn new_instance(&self, input: &Variable) -> Box<dyn Function> {
        Box::new(Square {
            input: Some(Variable::new(input.data)),
        })
    }
    fn forward(&self, input: f64) -> f64 {
        f64::powi(input, 2)
    }
    fn backward(&mut self, gy: f64) -> f64 {
        let x = self.input.as_ref().unwrap();
        2.0 * x.data * gy
    }
    fn get_input(&mut self) -> Option<&mut Variable> {
        self.input.as_mut()
    }
}

#[derive(Default)]
pub struct Exp {
    input: Option<Variable>,
}

impl Function for Exp {
    fn new_instance(&self, input: &Variable) -> Box<dyn Function> {
        Box::new(Exp {
            input: Some(Variable::new(input.data)),
        })
    }
    fn forward(&self, input: f64) -> f64 {
        f64::exp(input)
    }
    fn backward(&mut self, gy: f64) -> f64 {
        let x = self.input.as_ref().unwrap();
        f64::exp(x.data) * gy
    }
    fn get_input(&mut self) -> Option<&mut Variable> {
        self.input.as_mut()
    }
}

fn numerical_diff(f: &mut impl Function, x: Variable) -> f64 {
    let eps = 1e-4;
    let x0 = Variable::new(x.data - eps);
    let x1 = Variable::new(x.data + eps);
    let y0 = f.call(&x0);
    let y1 = f.call(&x1);
    return (y1.data - y0.data) / (eps * 2.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_test() {
        let x1 = 10.0;
        let expected = f64::powi(x1, 2);
        let x2 = Variable::new(10.0);
        let mut f = Square::default();
        let actual = f.call(&x2);
        assert_eq!(expected, actual.data);
    }

    #[test]
    fn backward_test() {
        let x = 10.0;
        let mut f = Square::default();
        let mut y = f.call(&Variable::new(x));
        y.grad = 1.0;
        let result = y.backward();
        assert_eq!(true, result.is_some());
        assert_eq!(x, result.unwrap().data);
        assert_eq!(2.0 * x, result.unwrap().grad)
    }
}
