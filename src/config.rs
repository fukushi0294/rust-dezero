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

#[macro_export]
macro_rules! no_grad {
    () => {
        pub struct NoGradContext {}

        impl NoGradContext {
            fn new() -> Self {
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

        std::hint::black_box(NoGradContext::new())
    };
}
