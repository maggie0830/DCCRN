def get_net_params():
    params = {}
    encoder_dim_start = 32
    params["encoder_channels"] = [
        1,
        encoder_dim_start,
        encoder_dim_start * 2,
        encoder_dim_start * 4,
        encoder_dim_start * 8,
        encoder_dim_start * 8,
        encoder_dim_start * 8
    ]
    params["encoder_kernel_sizes"] = [
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2)
    ]
    params["encoder_strides"] = [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1)
    ]
    params["encoder_paddings"] = [
        None,
        None,
        None,
        None,
        None,
        None
    ]
    # ----------lstm---------
    params["lstm_dim"] = 128
    params["lstm_layer_num"] = 2
    params["dense"] = 1280
    # --------decoder--------
    params["decoder_channels"] = [
        encoder_dim_start * 8,
        encoder_dim_start * 8,
        encoder_dim_start * 8,
        encoder_dim_start * 4,
        encoder_dim_start * 2,
        encoder_dim_start

    ]
    params["decoder_kernel_sizes"] = [
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2),
        (5, 2)
    ]
    params["decoder_strides"] = [
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1),
        (2, 1)
    ]
    params["decoder_paddings"] = [
        None,
        None,
        None,
        None,
        None,
        None
    ]
