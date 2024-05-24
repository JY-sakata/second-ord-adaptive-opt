
from optimizers import *




def get_optimizer(config, net):
    """
    Set the configuration for each optimiser
    """
    optimizers_name = ['sgd', 'sgdm','acclip', 'adam', 'sophia', 'sadh','ssdh','amsgrad', 'sgd_norm_clip', 'ssvp','asvp','zsvp', 'randomssdh', 'ssvp_sign', 'ssvp_cum_sign', 'adamw']
    opt_name = config['optimizer_name']
    if opt_name not in optimizers_name:
        raise RuntimeError('Not supported optimizers')
    
    if opt_name == 'sgd':
        optimizer = SGD(net.parameters(), lr=config['lr'], )
    elif opt_name == 'sgdm':
        optimizer = SGD(net.parameters(), lr=config['lr'], momentum=0.9, )
    elif opt_name == 'adam':
        optimizer = Adam(net.parameters(), lr=config['lr'], betas=(0.9, config['beta2']) )
    elif opt_name == 'adamw':
        optimizer = AdamW(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'],betas=(0.9, config['beta2']))
    elif opt_name == 'sophia':
        optimizer = Sophia(net.parameters(), lr=config['lr'], rho=config['rho'], update_period=config['update_period'],weight_decay=config['weight_decay'],betas=(0.9, config['beta2']) )
    elif opt_name == 'ssdh':
        optimizer = SSDH(net.parameters(), lr=config['lr'], rho=config['rho'],  update_period=config['update_period'],weight_decay=config['weight_decay'],betas=(0.9, config['beta2']) )
    elif opt_name == 'sadh':
        optimizer = SADH(net.parameters(), lr=config['lr'], rho=config['rho'], update_period=config['update_period'],weight_decay=config['weight_decay'],betas=(0.9, config['beta2']) )
    elif opt_name == 'amsgrad':
        optimizer = Adam(net.parameters(), lr=config['lr'], amsgrad=True)
    elif opt_name == 'ssvp':
        optimizer = SSVP(net.parameters(), lr=config['lr'], rho=config['rho'],  update_period=config['update_period'], sign=True, weight_decay=config['weight_decay'],betas=(0.9, config['beta2']))
    elif opt_name == 'ssvp_cum_sign':
        optimizer = SSVP(net.parameters(), lr=config['lr'], rho=config['rho'],  update_period=config['update_period'], sign=False, betas=(0.9, config['beta2']))
    elif opt_name == 'zsvp':
        optimizer = ZSVP(net.parameters(), lr=config['lr'], rho=config['rho'], update_period=config['update_period'], sign=False, betas=(0.9, config['beta2']))
    # elif opt_name == 'asvp':
    #     optimizer = ASVP(net.parameters(), lr=config['lr'], rho=config['rho'], update_period=config['update_period'], sign=True)

    return optimizer



def get_best_optimiser_config_resnet18_cifar10(config):
    """
    Get the best configuration for optimisers to train the ResNet18 after hyperparameter tuning
    """
    opt_name = config['optimizer_name']
    if opt_name == 'ssvp':
        config['beta2'] = 0.9902417663300304
        config['lr'] = 0.012904851354049134
        config['rho'] = 0.012603469080986496
    elif opt_name == 'sgdm':
        config['lr'] = 0.013779694301860598
    elif opt_name == 'ssdh':
        config['lr'] = 0.13277684691947564
        config['beta2'] = 0.9962831495943887
        config['rho'] = 0.012674411273821481
    elif opt_name == 'sadh':
        config['lr'] = 0.11359105429374000
        config['rho'] = 0.00268745510805206
        config['beta2'] = 0.995824597621633
    elif opt_name == 'sophia':
        config['lr'] = 0.019151648350265000
        config['beta2'] = 0.9852396751140580
        config['rho'] = 0.012696656762303400
    elif opt_name == 'adam':
        config['lr'] = 0.005167177473387094
        config['beta2'] = 0.9856019660982269
    
    return config

    ######## results without tuning beta2
    # if opt_name == 'adam':
    #     config['lr'] = 0.0009588677585533595
    # elif opt_name == 'sgdm':
    #     config['lr'] = 0.013779694301860598
    # elif opt_name == 'sophia':
    #     config['lr'] = 0.01407473555015365
    #     config['rho'] = 0.0050358157610116494
    # elif opt_name == 'ssdh':
    #     config['lr'] = 0.12885929299960772
    #     config['rho'] = 0.03866044919430927
    # elif opt_name == 'sadh':
    #     config['lr'] = 0.16008875277977824
    #     config['rho'] = 0.003109426170513824
    # elif opt_name == 'ssvp':
    #     config['lr'] = 0.015377033910754694
    #     config['rho'] = 0.019798064319733244  
    # elif opt_name == 'asvp':
    #     config['lr'] = 0.03188817391964956
    #     config['rho'] = 0.0049967129822934345





# def get_best_optimiser_config_transformer_ptb(config):
#     opt_name = config['optimizer_name']
#     if opt_name == 'adam':
#         config['lr'] = 0.0006336942753599376
#         config['beta2'] = 0.9972635707994034
#     elif opt_name == 'ssvp':
#         config['lr'] = 0.0118309492539123
#         config['rho'] = 0.06327786268404842
#         config['beta2'] = 0.997865546168291
#     elif opt_name == 'ssdh':
#         config['lr'] = 0.04560649678470467
#         config['rho'] = 0.04560649678470467
#         config['beta2'] = 0.9891704955407422
    

#     return config



def get_best_optimiser_config_transformer_ptb_bs128(config):
    """
    Get the best configuration for optimisers to train the transformer after hyperparameter tuning
    """
    opt_name = config['optimizer_name']
    if opt_name == 'adam':
        config['lr'] = 0.0006336942753599376
        config['beta2'] = 0.9972635707994034
    elif opt_name == 'ssvp':
        config['lr'] = 0.0118309492539123
        config['rho'] = 0.06327786268404842
        config['beta2'] = 0.997865546168291
    elif opt_name == 'ssdh':
        config['lr'] = 0.04560649678470467
        config['rho'] = 0.021130023799954212
        config['beta2'] = 0.9891704955407422
    elif opt_name == 'sophia':
        config['lr'] = 0.014990078063352274
        config['rho'] = 0.034267839044521936
        config['beta2'] = 0.9941346784703174
    elif opt_name == 'sadh':
        config['lr'] = 0.007854815465215976
        config['rho'] = 0.06927414239173069
        config['beta2'] = 0.9975434540675855
    elif opt_name == 'sgdm':
        config['lr'] = 0.04982403314514187

    return config
    

def get_best_optimiser_config_RR_cond29(config):
    opt_name = config['optimizer_name']
    if opt_name == 'adam':
        config['lr'] = 0.026939245909132378
        config['beta2'] = 0.9997256169023133
    elif opt_name == 'sadh':
        config['lr'] = 0.24374167324615045
        config['rho'] = 0.26242574017295567
        config['beta2'] = 0.9877327271968162
    elif opt_name == 'sgdm':
        config['lr'] = 0.04738368653444532
    elif opt_name == 'sophia':
        config['lr'] = 0.474135447805203
        config['rho'] = 0.02332056131872686
        config['beta2'] = 0.9926620931323629
    elif opt_name == 'ssdh':
        config['lr'] = 0.23406964650390694
        config['rho'] =0.18324428546251648
        config['beta2'] = 0.9870784585237155
    elif opt_name == 'ssvp':
        config['lr'] = 0.125236903033728
        config['rho'] = 0.0780816607160677
        config['beta2'] = 0.9939200114145359
    config['net_path'] = 'net_RR_cond29_xdim12_wdim13_ydim14_nlayers2.pth'

    
    return config


def get_best_optimiser_config_RR_cond15099(config):
    opt_name = config['optimizer_name']
    if opt_name == 'sophia':
        config['lr'] = 0.5100248140596734
        config['rho'] = 0.5
        config['beta2'] = 0.985
    elif opt_name == 'ssdh':
        config['lr'] = 0.2828646758735336
        config['beta2'] = 0.9999
        config['rho'] = 0.2912896416787145
    elif opt_name == 'ssvp':
        config['lr'] = 0.125236903033728
        config['beta2'] = 0.9939200114145359
        config['rho'] = 0.0780816607160677
    elif opt_name == 'adam':
        config['beta2'] = 0.9994677611330212
        config['lr'] = 0.033707979041547285
    elif opt_name == 'sadh':
        config['lr'] = 0.1702651189872818
        config['rho'] = 0.0909943011068296
        config['beta2'] = 0.9974033953479263
    elif opt_name == 'sgdm':
        config['lr'] = 0.04637126868083841
    
    return config

# def get_best_optimiser_config_transformer_ptb_bs256(config):
#     opt_name = config['optimizer_name']
#     if opt_name == 'adam':
#         config['lr'] = 0.0019227452899253298
#     elif opt_name == 'ssdh':
#         config['lr'] = 0.04136129453172596
#         config['rho'] = 0.006718389430823516
#     elif opt_name == 'ssvp':
#         config['lr'] = 0.029211801589900728
#         config['rho'] = 0.07576260787592246
#     elif opt_name == 'sadh':
#         config['lr'] = 0.04920949278975132
#         config['rho'] = 0.004802021558635069
#     elif opt_name == 'sophia':
#         config['lr'] = 0.0036028443785904697
#         config['rho'] = 0.07649133588909486

#     return config


# def get_best_optimiser_config_transformer_ptb_bs512(config):
#     opt_name = config['optimizer_name']
#     if opt_name == 'adam':
#         config['lr'] = 0.0020179889685721627
#     elif opt_name == 'ssdh':
#         config['lr'] = 0.0424075958549274
#         config['rho'] = 0.024417172067365458
#     elif opt_name == 'ssvp':
#         config['lr'] = 0.044228469202770966
#         config['rho'] = 0.06848409727223921
#     elif opt_name == 'sadh':
#         config['lr'] = 0.02130053869657682
#         config['rho'] = 0.022763957002082582
#     elif opt_name == 'sophia':
#         config['lr'] = 0.02178352221087173
#         config['rho'] = 0.017337103558513634
    

#     return config
