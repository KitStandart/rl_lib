from ..a2c.actor_critic import _abc_Actor_Model, _abc_Critric_Model, _abc_Actor_Critic_Model


class PPO_Actor_Model(_abc_Actor_Model):
    def __init__(self, config=..., **kwargs):
        super().__init__(config, **kwargs)
    
class PPO_Critic_Model(_abc_Critric_Model):
    def __init__(self, config=..., **kwargs):
        super().__init__(config, **kwargs)
    
class PPO_Model(_abc_Actor_Critic_Model):
    def __init__(self, actor_model: object = None, critic_model: object = None, config: dict = ..., **kwargs):
        super().__init__(actor_model, critic_model, config, **kwargs)