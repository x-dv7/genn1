use std::f32::consts::E as Eul;
//use std::fmt;

pub type NodeIndex = u16;
pub type LinkIndex = u32;

// /// Индекс (id) узла
// #[derive(Deserialize, Serialize, Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
// pub struct NodeId(NodeIndex);

// impl NodeId {
//     pub const MIN: usize = 0;
//     pub const MAX: usize = NodeIndex::MAX as usize;

//     pub fn new(index: usize) -> Self {
//         if index > Self::MAX as usize {
//             panic!("NodeId too small, layer has more then {} nodes", Self::MAX);
//         }
//         Self(index as NodeIndex)
//     }

//     pub fn index(&self) -> usize {
//         self.0 as usize
//     }
// }

// impl fmt::Display for NodeId {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{:?}", self)
//     }
// }

// /// Индекс (id) связи
// #[derive(Deserialize, Serialize, Debug, Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
// pub struct LinkId(LinkIndex);

// impl LinkId {
//     pub const MIN: usize = 0;
//     pub const MAX: usize = LinkIndex::MAX as usize;

//     pub fn new(index: usize) -> Self {
//         if index > Self::MAX as usize {
//             panic!("LinkId too small, layer has more then {} links", Self::MAX);
//         }
//         Self(index as LinkIndex)
//     }

//     pub fn index(&self) -> usize {
//         self.0 as usize
//     }
// }

// impl fmt::Display for LinkId {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{:?}", self)
//     }
// }

/// Тип узла
#[derive(Deserialize, Serialize, Debug, PartialEq, Clone, Copy)]
pub enum NodeType {
    Sensor,//входной узел
    Output,//выходной узел
    Hidden//внутренний узел
}

/// функции активации для нейронной сети, должны быть указаны при создании
#[derive(Deserialize, Serialize, Debug, PartialEq, Clone, Copy)]
pub enum Activation {
    Sigmoid,
    Tanh,
    Relu,
    Softmax,
    LeakyRelu(f32),
    ExpRelu(f32),
    Linear(f32)   
}


impl Activation {

    /// функции активации для нейронной сети/некоторым нужен параметр alpha
    #[inline]
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            Self::Sigmoid => {
                1.0 / (1.0 + (-x * 4.9).exp())
            },
            Self::Tanh => {
                x.tanh()
            },
            Self::Relu => { 
                if x > 0.0 { 
                    x 
                } else { 
                    0.0 
                }
            },
            Self::Linear(alpha) => {
                alpha * x
            },
            Self::LeakyRelu(alpha) => {
                let a = alpha * x;
                if a > x {
                    return a;
                } 
                x
            },
            Self::ExpRelu(alpha) => {
                if x >= 0.0 {
                    return x;
                }
                alpha * (Eul.powf(x) - 1.0)
            },
            _ => panic!("Cannot activate single neuron")

        }
    }
}

// // (код на Rust)
// trait Distance {
// 	fn distance(&self, other: &Self) -> f64;
// }

// // для чисел
// impl Distance for f64 {
// 	fn distance(&self, other: &Self) -> f64 {
// 		return (self - other).abs();
// 	}
// }


// // для векторов и массивов
// impl Distance for Vec<f64> {
// 	fn distance(&self, other: &Self) -> f64 {
// 		let mut s = 0.0;
//     let n = self.len();
    
// 		for i in 0..n {
// 			let x = self[i] - other[i];
// 			s += x*x;
// 		}
// 		s.sqrt()
// 	}
// }


// struct BWImage {
//   matrix : Vec<Vec<u8>>,
// }

// // для чёрно-белых картинок
// impl Distance for BWImage {
// 	fn distance(&self, other: &Self) -> f64 {
// 		let mut s = 0.0;
//     let n = self.len();
//     let m = self[0].len();
    
//     for i in 0..n {
//       for j in 0..m {
//         let x = self.matrix[i][j] - other.matrix[i][j];
//         s += x*x;
//       }
//     }
//     s.sqrt()
// 	}
// }


// // for strings
// impl Distance for String {
// 	fn distance(&self, other: &Self) -> f64 {
// 		return (self == other) as i32 as f64;
// 	}
// }