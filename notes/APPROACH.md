# Approach

# Target Baseline Approach

```
11 	Pseudo17 	Nspec23arch3 	CNN 	efficientnet-b6 	128 x 1024 	0.87982 	0.8823 	0.8808
```

DL Model Pipeline:

```
CNNSpectrogram --> Resizing --> `tf_efficientnet_b6_ns`
```

---

```
class Baseline:
    name = 'baseline'
    seed = 2021
    train_path = INPUT_DIR/'train.csv'
    test_path = INPUT_DIR/'test.csv'
    train_cache = None # You can add the path to your dataset cache
    test_cache = None  #
    cv = 5
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    dataset = G2NetDataset
    dataset_params = dict()

    model = SpectroCNN
    model_params = dict(
        model_name='tf_efficientnet_b7',
        pretrained=True,
        num_classes=1,
        spectrogram=CQT,
        spec_params=dict(sr=2048,
                         fmin=20,
                         fmax=1024,
                         hop_length=64),
    )
    weight_path = None
    num_epochs = 5
    batch_size = 64
    optimizer = optim.Adam
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts
    scheduler_params = dict(T_0=5, T_mult=1, eta_min=1e-6)
    scheduler_target = None
    batch_scheduler = False
    criterion = nn.BCEWithLogitsLoss()
    eval_metric = AUC().torch
    monitor_metrics = []
    amp = True
    parallel = None
    deterministic = False
    clip_grad = 'value'
    max_grad_norm = 10000
    hook = TrainHook()
    callbacks = [
        EarlyStopping(patience=5, maximize=True),
        SaveSnapshot()
    ]

    transforms = dict(
        train=None,
        test=None,
        tta=None
    )

    pseudo_labels = None
    debug = False
```

```
class Resized08aug4(Baseline):
    name = 'resized_08_aug_4'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=CQT,
        spec_params=dict(sr=2048,
                         fmin=16,
                         fmax=1024,
                         hop_length=8),
        resize_img=(256, 512),
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.25)
        ]),
        test=None,
        tta=None
    )
    dataset_params = dict(
        norm_factor=[4.61e-20, 4.23e-20, 1.11e-20])
    num_epochs = 8
    scheduler_params = dict(T_0=8, T_mult=1, eta_min=1e-6)
    optimizer_params = dict(lr=1e-3, weight_decay=1e-6)
```

```
class Nspec22(Resized08aug4):
    name = 'nspec_22'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=WaveNetSpectrogram,
        spec_params=dict(
            base_filters=128,
            wave_layers=(10, 6, 2),
            kernel_size=3,
        ),
        resize_img=None,
        custom_classifier='gem',
        upsample='bicubic'
    )
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
    )
    dataset_params = dict()
```

```
class Nspec23(Nspec22):
    name = 'nspec_23'
    model_params = dict(
        model_name='tf_efficientnet_b2',
        pretrained=True,
        num_classes=1,
        spectrogram=CNNSpectrogram,
        spec_params=dict(
            base_filters=128,
            kernel_sizes=(32, 16, 4),
        ),
        resize_img=None,
        custom_classifier='gem',
        upsample='bicubic'
    )
```

```
class Nspec23arch3(Nspec23):
    name = 'nspec_23_arch_3'
    model_params = Nspec23.model_params.copy()
    model_params['spec_params'] = dict(
        base_filters=128,
        kernel_sizes=(64, 16, 4),
    )
    model_params['model_name'] = 'tf_efficientnet_b6_ns'
    transforms = Nspec22aug1.transforms.copy()
```

## Possible Baseline Approaches

Use these configs:

Located @ https://github.com/analokmaus/kaggle-g2net-public/blob/main/configs.py.

```
11 	Pseudo17 	Nspec23arch3 	CNN 	efficientnet-b6 	128 x 1024 	0.87982 	0.8823 	0.8808
12 	Pseudo21 	Nspec22arch7 	WaveNet 	effnetv2-m 	128 x 1024 	0.87861 	0.8831 	0.8815
```

```
class Nspec22arch7(Nspec22):
    name = 'nspec_22_arch_7'
    model_params = Nspec22aug1.model_params.copy()
    model_params['model_name'] = 'tf_efficientnetv2_m'
    transforms = Nspec22aug1.transforms.copy()
    optimizer_params = dict(lr=2e-4, weight_decay=1e-6)
```
