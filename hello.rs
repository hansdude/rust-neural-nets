use std::f64;

struct Neuron {
    weights: Vec<f64>,
}

struct Entry {
    inputs: Vec<f64>,
    outputs: Vec<f64>
}

fn sigmoid(x: f64) -> f64 {
    //let c = 1.0;
    //1.0 / (1.0 + f64::exp(-c * x))
    x
}
fn _sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn main() {
    // XOR function
    let training = vec![
        Entry {
            inputs: vec![0.0, 0.0],
            outputs: vec![0.0],
        },
        Entry {
            inputs: vec![0.0, 1.0],
            outputs: vec![1.0],
        },
        Entry {
            inputs: vec![1.0, 0.0],
            outputs: vec![1.0],
        },
        Entry {
            inputs: vec![1.0, 1.0],
            outputs: vec![0.0],
        },
    ];

    let layers = vec![
        vec![
            Neuron { weights: vec![0.0, 1.0, 1.0] },
            Neuron { weights: vec![0.0, 1.0, 1.0] },
        ],
        vec![
            Neuron { weights: vec![0.0, 1.0, 1.0] },
        ],
    ];

    let input = vec![0.25, 0.25];
    let output = forward(input, &layers);

    println!("output");
    for v in output {
        println!("{}", v);
    }
}

fn forward(input: Vec<f64>, layers: &Vec<Vec<Neuron>>) -> Vec<f64> {
    layers.iter().fold(input, forward_layer)
}

fn forward_layer(input: Vec<f64>, layer: &Vec<Neuron>) -> Vec<f64> {
    layer.iter().map(|neuron| {
        sigmoid(
            neuron.weights.iter()
                .skip(1)
                .zip(&input)
                .map(|(x, y)| x * y)
                .fold(neuron.weights[0], |x, y| x + y)
        )
    }).collect()
}

