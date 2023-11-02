import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = 'SACLearner'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.num_qs = 2

    config.critic_dropout_rate = 0.01
    config.critic_layer_norm = True

    config.tau = 0.005
    config.init_temperature = 0.1
    config.target_entropy = config_dict.placeholder(float)
    
    # config.ctrl_weight = 0.0

    return config