import torch


def get_caption(model, image):
    vocab = "abcdefghijklmnopqrstuvwxyz "
    vocab = dict(zip(list(vocab), range(len(vocab))))
    vocab["<start>"] = 27
    vocab["<end>"] = 28
    vocab_opp = {vocab[key]: key for key in vocab}

    chars = ""
    inp = torch.zeros((1, 1, 29)).to(model.device)
    inp[0, 0, 27] = 1  # <start>
    idx = None

    while idx != 28:  # <end>
        with torch.no_grad():
            logits = model.forward(inp, image)

        idx = torch.argmax(logits[-1]).item()
        chars += vocab_opp[idx]

        x = torch.zeros((1, 1, 29)).to(model.device)
        x[0, 0, idx] = 1
        inp = torch.cat((inp, x), dim=0)

    return chars[:-5]
