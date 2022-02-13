

/// Node - узел сети
#[derive(Deserialize, Serialize, Debug)]
pub struct Node {
    pub id: NodeIndex,//исторический номер узла
    incoming: Vec<NodeLink>,//список связей смотрящих на узел
    activation: Activation,
    direction: NeuronDirection,
    pub node_type: NodeType,
    pub activated_value: f32,
    pub deactivated_value: f32,
    pub current_state: f32,
    pub previous_state: f32,
    pub error: f32,
    pub bias: f32,
}


#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NodeLink {
    pub id: LinkIndex,//исторический номер связи
    pub src: NodeIndex,//откуда связь
    pub dst: NodeIndex,//куда связь
    pub weight: f32,//вес связи
    pub enable: bool,//включена?
}

