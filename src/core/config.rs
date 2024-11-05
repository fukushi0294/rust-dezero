use once_cell::sync::Lazy;
use std::sync::Mutex;

pub struct Config {
    pub enable_backprop: bool,
}

impl Config {
    fn new() -> Self {
        Config {
            enable_backprop: true,
        }
    }
}

pub static CONFIG: Lazy<Mutex<Config>> = Lazy::new(|| Mutex::new(Config::new()));

pub struct NoGradContext {}

impl NoGradContext {
    pub fn new() -> Self {
        {
            let mut config = CONFIG.lock().unwrap();
            config.enable_backprop = false;
        }
        NoGradContext {}
    }
}

impl Drop for NoGradContext {
    fn drop(&mut self) {
        let mut config = CONFIG.lock().unwrap();
        config.enable_backprop = true
    }
}

#[macro_export]
macro_rules! no_grad {
    {$($code:tt)*} => {
        std::hint::black_box($crate::core::config::NoGradContext::new());
        $($code)*
    };
}
