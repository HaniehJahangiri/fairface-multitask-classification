def get_model():
    model = models.vgg16(pretrained = True)
    model_path = 'vgg16_model.pth'
    model.load_state_dict(torch.load(model_path , map_location = device))
    for params in model.parameters():
        params.requires_grad = False
    model.avgpool = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=(3, 3)),  # Reduced from (7,7) to (3,3)
        nn.Conv2d(512, 256, kernel_size=1),  # Bottleneck: reduce channels
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.2),
        nn.Flatten()
    )
    class Age_Gender_Race_Classifier(nn.Module):
        def __init__(self):
            super(Age_Gender_Race_Classifier,self).__init__()
            self.intermediate = nn.Sequential(
                nn.Linear(256*3*3, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(.4),
                nn.Linear(768,384),
                nn.BatchNorm1d(384),
                nn.ReLU(),
                nn.Dropout(.3),
                nn.Linear(384,192),
                nn.ReLU(),
            )
            self.AgeClassifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
                
            )
            self.RaceClassifier = nn.Sequential(
            nn.Linear(192, 256),  
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 7)
            )
            self.GenderClassifier = nn.Sequential(
            nn.Linear(192, 48),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(48, 1),
            nn.Sigmoid()
            )
        def forward(self,x):
            x = self.intermediate(x)
            age = self.AgeClassifier(x)
            race = self.RaceClassifier(x)
            gender = self.GenderClassifier(x)
            return age,gender,race
    model.classifier = Age_Gender_Race_Classifier()
    gender_criterion = nn.BCELoss()
    age_criterion = nn.L1Loss()
    race_criterion = nn.CrossEntropyLoss(label_smoothing=.1)
    def weighted_loss(age_pred, gender_pred, race_pred, age_true, gender_true, race_true):
        age_loss = age_criterion(age_pred.squeeze(),age_true)
        gender_loss = gender_criterion(gender_pred.squeeze(),gender_true)
        race_loss = race_criterion(race_pred,race_true)
        total_loss = age_loss*.2 + gender_loss*.1 + race_loss*.7
        return total_loss,age_loss,gender_loss,race_loss
    loss_function = weighted_loss
    optimizer = torch.optim.Adam(model.classifier.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    return model.to(device),loss_function,optimizer,scheduler