
pub use super::base::*;
use rand::Rng;
use std::borrow::BorrowMut;
//use std::io::Read;
use std::sync::{Arc, Weak, RwLock};
use std::cell::RefCell;
use std::collections::{HashMap,HashSet};

/// NEAT сеть, состоящая из нескольких подсетей, связанных послойно
#[derive(Deserialize, Serialize, Debug)]
pub struct Nets {
    pub max_node: RwLock<NodeIndex>,
    //pub nodes: RwLock<HashSet<NodeIndex>>,//65536 узлов
    pub max_link: RwLock<LinkIndex>,
    //pub links: RwLock<HashSet<LinkIndex>>,//4 млрд связей
    pub layers: RwLock<Vec<DenseNet>>,//все сети 
}

impl Nets {
    pub fn new() -> Arc<Self> {
        Arc::new(
        Nets {
            max_node: RwLock::new(0),
            //nodes: RwLock::new(HashSet::new()),
            max_link: RwLock::new(0),
            //links: RwLock::new(HashSet::new()),
            layers: RwLock::new(Vec::new()),
        })
    }
}

/// Полносвязанная подсеть
#[derive(Deserialize, Serialize, Debug)]
pub struct DenseNet {
    pub parent_net: Weak<Nets>,//ссылка на родительскую NEAT-сеть
    pub nodes: HashMap<NodeIndex, DenseNode>,//узлы подсети
    pub links: HashMap<LinkIndex, NodesLink>,//связи между узлами
}

impl DenseNet {
    pub fn new(parent: Arc<Nets>) -> Self {
        DenseNet {
            parent_net: Arc::downgrade(&parent),
            nodes: HashMap::new(),
            links: HashMap::new(),
        }
    }

    /// добавить новый узел (для нового узла пишем id = 0)
    pub fn ins_node(&mut self, id: NodeIndex, node_type: NodeType, activation: Activation) -> NodeIndex {
        let mut lv_max_node: NodeIndex = 0;
        let lo_dense_net = Arc::new(*self);//ссылка на себя как на родителя
        if id == 0 {
            // добавляем новый узел. id указываем = 0. инкрементируем max_node
            if let Some(parent) = self.parent_net.upgrade() {
                if let Ok(max_node) = parent.max_node.write() {
                    *max_node += 1;
                    lv_max_node = *max_node;
                    let lo_net = DenseNode::new(lv_max_node, lo_dense_net, node_type, activation);
                    self.nodes.insert(lv_max_node, lo_net); 
                }
            } 
            lv_max_node
        }
        else 
        if let Some(_nid) = self.nodes.iter().find(|nid| *nid.0 == id) {
            //такой узел уже есть, возвращаем 0;
            0
        } else  {
            //такого узла не найдено, добавим новый с указанным NodeId
            if let Some(parent) = self.parent_net.upgrade() {
                if let Ok(max_node) = parent.max_node.write() {
                    if *max_node < id {*max_node = id};
                    lv_max_node = id;
                    

                }
            }
            if self.max_node.index() < id.index() { self.max_node = id };
            let n = Arc::new(RwLock::new(Node::new(id, node_type, activation)));
            self.nodes.insert(id, Arc::clone(&n));  
            id  
        }
    }
}
    // /// связать узлы (src - откуда, dst - куда). перед связыванием надо проверять на возможность!
    // pub fn link_nodes(&mut self, src: NodeId, dst: NodeId, weight: f32, enabled: bool) -> LinkId {
    //     //инкрементируем max_link
    //     self.max_link = LinkId::new(self.max_link.index() + 1);
    //     //создадим связь
    //     let lo_link = NodesLink::new(self.max_link, src, dst, weight, enabled);
    //     self.links.insert(self.max_link, Arc::new(RwLock::new(lo_link)));
    //     self.max_link
    // }

    // /// проверим, можно ли связать узлы (такие узлы должны существовать и связи между ними не должно быть)
    // pub fn check_before_link_nodes(&self, src: NodeId, dst: NodeId) -> bool {
    //     //сначала найдем эти узлы
    //     if let Some(_src) = self.nodes.iter().find(|n_id| n_id.0.index() == src.index()) {
    //         if let Some(_dst) = self.nodes.iter().find(|n_id| n_id.0.index() == dst.index()) {
    //             //проверим, вдруг такая связь уже есть:
    //             //1) в исх. связях src узла ищем ссылается ли хоть одна связь на dst
    //             let mut lv_res = false;
    //             if let Ok(s) = _src.1.read() {
    //                 if let Some(lo_link) = s.outgoing.iter().find(|l_id|{
    //                     if let Some(u) = l_id.upgrade() {
    //                         if let Ok(r) = u.read() {
    //                             if r.dst.index() == dst.index() {lv_res = true}
    //                         }
    //                     }
    //                     lv_res
    //                 }) {return false;};
    //             };
    //             if let Ok(s) = _dst.1.read() {
    //                 let r = s.incoming.iter().find(predicate)
    //             };
    //             //let r = _dst.1.try_read().ok().unwrap_or_default().outgoing.iter().unzip()
    //             //
    //             //s.incoming.iter().find(|n_id| {            
    //                 // if let Some(u) = *n_id.upgrade() {
    //                 //     if let Ok(r) = u.read() {
    //                 //         if r.id != link_id {true} else {false}
    //                 //     }
    //                 //     else {true}//сохраняем
    //                 // }
    //                 // else { true }//сохраняем
    //             //});
    //             //self.links.iter().find()
    //         }
            
    //     }
    //     true
    // }

    // /// Соединяем src/dst узлы
    // pub fn link_nodes(&self, nodes: &mut [Node]) {
    //     //nodes.get_mut(self.src.index()).map(|x| x.add_outgoing(self.id));
    //     nodes.get_mut(self.dst.index()).map(|x| x.add_incoming(self));
    // }

    // /// Enable edge and link the nodes.
    // pub fn enable(&mut self, nodes: &mut [Node]) {
    //     if self.enabled {
    //         // already active, nothing to do.
    //         return;
    //     }
    //     self.enabled = true;
    //     //nodes.get_mut(self.src.index()).map(|x| x.add_outgoing(self.id));
    //     // For dst node, just re-enable the weight.
    //     // This allows for faster forward propagation.
    //     nodes.get_mut(self.dst.index()).map(|x| x.update_incoming(self, self.weight));
    // }

    // /// Disable edge and unlink the nodes.
    // pub fn disable(&mut self, nodes: &mut [Node]) {
    //     self.enabled = false;
    //     //nodes.get_mut(self.src.index()).map(|x| x.remove_outgoing(self.id));
    //     // For dst node, just set the weight to zero.
    //     // This allows for faster forward propagation.
    //     nodes.get_mut(self.dst.index()).map(|x| x.update_incoming(self, 0.0));
    // }

// }

/// Node - узел сети
#[derive(Deserialize, Serialize, Debug)]
pub struct DenseNode {
    pub id: NodeIndex,//исторический индекс узла
    //outgoing: Vec<Weak<RwLock<NodesLink>>>,//список связей смотрящих из узла
    parent_net: Weak<DenseNet>,//ссылка на родительскую подсеть
    incoming: Vec<LinkIndex>,//связи, смотрящие на узел (индексы связей из родительской подсети DenseNet.links)
    activation: Activation,//функция активации
    pub node_type: NodeType,//тип узла
    pub activated_value: f32,//значение после активации
    //pub deactivated_value: f32,
    pub current_state: f32,//значение до активации
    //pub previous_state: f32,
    //pub error: f32,//ошибка обучения
    pub bias: f32,//смещение
    pub enabled: bool,//включен?
}

impl DenseNode {
    pub fn new(id: NodeIndex, parent: Arc<DenseNet>, node_type: NodeType, activation: Activation) -> Self {
        DenseNode {
            id,
            //outgoing: Vec::new(),
            parent_net: Arc::downgrade(&parent),
            incoming: Vec::new(),
            activation,
            node_type,
            activated_value: 0.0,
            //deactivated_value: 0.0,
            current_state: 0.0,
            //previous_state: 0.0,
            //error: 0.0,
            bias: rand::thread_rng().gen::<f32>(),
            enabled: true,
        }
    }

    /// добавляем связь на нас
    pub fn add_incoming(&mut self, link: LinkIndex) {
        self.incoming.push(link);
    }

    // /// добавляем связь от нас
    // pub fn add_outgoing(&mut self, link: Arc<RwLock<NodesLink>>) {
    //     self.outgoing.push(Arc::downgrade(&link));
    // }
    
    /// обновляем связь на нас
    pub fn update_incoming(&mut self, link: &NodesLink, weight: f32) {
        if let Some(link) = self.incoming.iter_mut().find(|x| x.id == link.id) {
            link.weight = weight;
        }
    }

    /// удаляем связь на нас
    pub fn remove_incoming(&mut self, link_id: LinkIndex) {
        //оставляем то, что true
        self.incoming.retain(|x| x.id != link_id );
    }
    /// Получаем список связей.
    pub fn incoming_edges(&self) -> &[NodesLink] {
        &self.incoming
    }

    /// 𝜎(Σ(w * i) + b)
    /// Активируем значение current_state
    #[inline]
    pub fn activate(&mut self) {
        if self.activation != Activation::Softmax {
            self.activated_value = self.activation.activate(self.current_state);
        }
    }

    /// Сброс значений узла
    #[inline]
    pub fn reset_node(&mut self) {
        //self.error = 0.0;
        self.activated_value = 0.0;
        self.current_state = 0.0;
    }

    /// Клонирование узла
    #[inline]
    pub fn clone_with_values(&self) -> Self {
        DenseNode {
            id: self.id,
            //outgoing: self.outgoing.clone(),
            incoming: self.incoming.clone(),
            current_state: self.current_state.clone(),
            //previous_state: self.previous_state.clone(),
            activated_value: self.activated_value.clone(),
            //error: self.error.clone(),
            bias: self.bias.clone(),
            activation: self.activation.clone(),
            node_type: self.node_type.clone(),
            enabled: self.enabled.clone()
        }
    }
}

impl Clone for DenseNode {
    fn clone(&self) -> Self { 
        DenseNode {
            id: self.id,
            //outgoing: self.outgoing.clone(),
            incoming: self.incoming.clone(),
            current_state: 0.0,
            //previous_state: 0.0,
            activated_value: 0.0,
            //deactivated_value: 0.0,
            //error: 0.0,
            bias: self.bias.clone(),
            activation: self.activation.clone(),
            node_type: self.node_type.clone(),
            enabled: self.enabled.clone()
        }
    }
}

/// Link - связь между 2-мя узлами сети, находящаяся в общем списке связей структуры Node
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NodesLink {
    pub id: LinkIndex,//исторический индекс связи
    pub src: NodeIndex,//откуда связь
    pub dst: NodeIndex,//куда связь
    pub weight: f32,//вес связи
    pub enabled: bool,//включена?
}

impl NodesLink {
    pub fn new(id: LinkIndex, src: NodeIndex, dst: NodeIndex, weight: f32, enabled: bool) -> Self {
        NodesLink {
            id,
            src,
            dst,
            weight,
            enabled
        }
    }

    /// Обновление веса этого соединения на дельту
    #[inline]
    pub fn update(&mut self, delta: f32/*, nodes: &mut [Node]*/) {
        self.update_weight(self.weight + delta/*, nodes*/);
    }

    /// Обновление непосредственно значения веса
    pub fn update_weight(&mut self, weight: f32 /*, nodes: &mut [Node]*/) {
        self.weight = weight;
        //непонятный кусок кода!!!
        //nodes.get_mut(self.dst.index()).map(|x| x.update_incoming(self, weight));
    }
    
    /// вычисление (w * i), где i - вх.значение
    #[inline]
    pub fn calculate(&self, i_val: f32) -> f32 {
        self.weight * i_val //(w * i)
    }
}

// /// SymbolLink - обозначение связи между 2-мя узлами сети, находящаяся в структуре Node.
// /// Сделано так, чтобы не лепить сильные и слабые указатели, а определять связь по её индексу
// #[derive(Deserialize, Serialize, Debug, Clone)]
// pub struct SymbolLink {
//     pub id: LinkIndex,//исторический индекс связи
//     pub src: NodeIndex,//индекс узла, откуда идет связь
//     pub weight: f32,//вес связи
// }

// impl SymbolLink {
//     pub fn new(link: &NodesLink) -> Self {
//         Self {
//             id: link.id,
//             src: link.src,
//             weight: link.weight,
//         }
//     }
// }