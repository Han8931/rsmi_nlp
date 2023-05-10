
def load_base_model(args):
    print(f"Load Model...")
    if args.model == 'bert':
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.num_classes)
    elif args.model == 'bert-large':
        from transformers import BertForSequenceClassification 
        model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=args.num_classes)
    elif args.model == 'roberta':
        from transformers import RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=args.num_classes)
    elif args.model == 'roberta-large':
        from transformers import RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=args.num_classes)
    else:
        raise Exception("Specify model correctly...")
    model.config.num_labels = args.num_classes
    return model

def load_tokenizer(args):
    print(f"Load Tokenizer...")
    if args.model == 'bert':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.model == 'bert-large':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    elif args.model == 'roberta':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif args.model == 'roberta-large':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    else:
        raise Exception("Specify model correctly...")

    args.pad_idx = tokenizer.pad_token_id
    args.mask_idx = tokenizer.mask_token_id
    print(f"Tokenizer: {args.model} || PAD: {args.pad_idx} || MASK: {args.mask_idx}") 
    return tokenizer

def noisy_forward_loader(args):
    if 'roberta' in args.model:
        from transformers import RobertaForSequenceClassification

        if args.model == 'roberta-large':
            model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=args.num_classes)
        else:
            model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=args.num_classes)

        import types
        from model.model_noise_forward import roberta_noise_forward 
        model.roberta.encoder.forward = types.MethodType(roberta_noise_forward, model.roberta.encoder)

    elif 'bert' in args.model:
        from transformers import BertForSequenceClassification

        if args.model == 'bert-large':
            model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=args.num_classes)
        else:
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.num_classes)

        import types
        from model.model_noise_forward import bert_noise_forward
        model.bert.encoder.forward = types.MethodType(bert_noise_forward, model.bert.encoder)

    else:
        raise Exception("Specify Base model correctly...")

    model.config.single_layer = args.single_layer
    model.config.num_labels = args.num_classes
    model.config.nth_layers = args.nth_layers
    model.config.noise_eps = args.noise_eps

    return model
