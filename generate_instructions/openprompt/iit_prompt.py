# step1: 确定NLP任务
# -*- encoding:utf-8 -*-
import torch
from random import choice
from openprompt.data_utils import InputExample
from ..utils import InputData

def predict_affordance(c, box):
    # IIT dataset classes
    classes = [
        "contain",
        "cut",
        "display",
        "engine",
        "hit",
        "support",
        "pound"
    ]

    # Explicit template
    explicit_templates = [
        "I need the {}",
        "Hand me the {}",
        "Pass me the {}",
        "I want the {}",
        "Bring me the {}",
        "I want to use the {}",
        "Get the {}",
        "Give me the {}",
        "Fetch the {}",
        "Bring the {}",
    ]

    # Implicit template
    implicit_templates = [
        'An item that can',
        'An object that can',
        'Give me something that can',
        'Give me an item that can',
        'Hand me something to',
        'Give me something to',
        'I want something to',
        'I need something to',
    ]
    i = 0
    dataset = []
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    for c in c:
        dataset.append(
            InputExample(
                guid=i,
                text_a=choice(explicit_templates).format(c),
                text_b=c,
                meta={
                    "sentence": choice(explicit_templates).format(c),
                    "entity": c
                }
            ),
        )
        i = i + 1

    # step 2 Define a pre-trained language model (PLMs) as the backbone.
    from openprompt.plms import load_plm
    # plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    plm, tokenizer, model_config, WrapperClass = load_plm("gpt2", "gpt2")
    # step 3
    from openprompt.prompts import ManualTemplate
    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"}.In other words, give me something to {"mask"}',
        tokenizer=tokenizer,
    )
    # step 4
    # Verbalizer projects the original label into a set of lable words.
    from openprompt.prompts import ManualVerbalizer

    label_words = {
        "contain": ["contain", "drink", "pour", "cook"],
        "cut": ["cut"],
        "display": ["show", "observe", "watch"],
        "engine": ["engine", "operate"],
        "pound": ["hit", "strike", "beat"],
        # "support": ["smooth", "stir fry", "support"],
        "hit": ["swing", "play"]
    }

    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words=label_words,
        tokenizer=tokenizer,
    )
    # step 5 Merged into PromptModel
    from openprompt import PromptForClassification
    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )
    # step 6
    from openprompt import PromptDataLoader
    data_loader = PromptDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
    )
    # step 7
    # making zero-shot inference using pretrained MLM with prompt

    promptModel.eval()
    dataset1 = []
    with torch.no_grad():
        for batch in data_loader:
            logits = promptModel(batch)
            preds = torch.argmax(logits, dim=-1)
            print(classes[preds])
            current_box = box[batch.guid.item()]
            location_length = len(current_box[4:])
            if location_length == 1:
                location_1 = dataset[batch.guid.item()].text_a + ' on the ' + current_box[4] + ' to '
                grasp_location_1 = ' on the ' + current_box[4]
            elif location_length == 2:
                location_2 = dataset[batch.guid.item()].text_a + ' on the ' + current_box[4] + ' and ' + current_box[5] + ' to '
                grasp_location_2 = ' on the ' + current_box[4] + ' and ' + current_box[5]
            elif location_length == 3:
                location_3 = dataset[batch.guid.item()].text_a + ' on the ' + current_box[4] + ',' + current_box[5] + ' and ' + current_box[6] + ' to '
                grasp_location_3 = ' on the ' + current_box[4] + ',' + current_box[5] + ' and ' + current_box[6]
            if classes[preds] == "contain":
                if dataset[batch.guid.item()].text_b == "pan":
                    implicit_sentence = choice(implicit_templates) + ' ' + choice(["contain", "cook"])
                    if location_length == 1:
                        explicit_sentence0 = location_1 + choice(["contain", "cook"])
                    elif location_length == 2:
                        explicit_sentence0 = location_2 + choice(["contain", "cook"])
                    elif location_length == 3:
                        explicit_sentence0 = location_3 + choice(["contain", "cook"])
                    else:
                        explicit_sentence0 = dataset[batch.guid.item()].text_a + ' to ' + choice(["contain", "cook"])
                else:
                    implicit_sentence = choice(implicit_templates) + ' ' + choice(["contain", "drink", "pour"])
                    if location_length == 1:
                        explicit_sentence0 = location_1 + choice(["contain", "drink", "pour"])
                    elif location_length == 2:
                        explicit_sentence0 = location_2 + choice(["contain", "drink", "pour"])
                    elif location_length == 3:
                        explicit_sentence0 = location_3 + choice(
                            ["contain", "drink", "pour"])
                    else:
                        explicit_sentence0 = dataset[batch.guid.item()].text_a + ' to ' + choice(
                            ["contain", "drink", "pour"])

            else:
                implicit_sentence = choice(implicit_templates) + ' ' + choice(label_words[classes[preds]])
                if location_length == 1:
                    explicit_sentence0 = location_1 + choice(label_words[classes[preds]])
                elif location_length == 2:
                    explicit_sentence0 = location_2 + choice(label_words[classes[preds]])
                elif location_length == 3:
                    explicit_sentence0 = location_3 + choice(label_words[classes[preds]])
                else:
                    explicit_sentence0 = dataset[batch.guid.item()].text_a + ' to ' + choice(label_words[classes[preds]])
            if dataset[batch.guid.item()].text_b == "bowl" or dataset[batch.guid.item()].text_b == "cup":
                if location_length == 1:
                    explicit_sentence1 = "W-grasp the " + dataset[batch.guid.item()].text_b + grasp_location_1
                elif location_length == 2:
                    explicit_sentence1 = "W-grasp the " + dataset[batch.guid.item()].text_b + grasp_location_2
                elif location_length == 3:
                    explicit_sentence1 = "W-grasp the " + dataset[batch.guid.item()].text_b + grasp_location_3
                else:
                    explicit_sentence1 = "W-grasp the " + dataset[batch.guid.item()].text_b
            elif dataset[batch.guid.item()].text_b != "tvm":
                if location_length == 1:
                    explicit_sentence1 = "Grasp the " + dataset[batch.guid.item()].text_b + grasp_location_1
                elif location_length == 2:
                    explicit_sentence1 = "Grasp the " + dataset[batch.guid.item()].text_b + grasp_location_2
                elif location_length == 3:
                    explicit_sentence1 = "Grasp the " + dataset[batch.guid.item()].text_b + grasp_location_3
                else:
                    explicit_sentence1 = "Grasp the " + dataset[batch.guid.item()].text_b
            else:
                explicit_sentence1 = ''

            dataset1.append(
                InputData(
                    guid=batch.guid.item(),
                    label=dataset[batch.guid.item()].text_b,
                    explicit_sentence=[explicit_sentence0, explicit_sentence1],
                    implicit_sentence=implicit_sentence,
                    box=current_box[0: 4]
                ),
            )


    return dataset1