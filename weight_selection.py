import os, argparse
import torch, timm
from models.vision_transformer import vit_tiny
from models.convnext import convnext_femto


def uniform_element_selection(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        # determine whether teacher is larger than student on this dimension
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"
        if wt.shape[dim] % s_shape[dim] == 0:
            step = wt.shape[dim] // s_shape[dim]
            indices = torch.arange(s_shape[dim]) * step
        else:
            indices = torch.round(torch.linspace(0, wt.shape[dim], s_shape[dim]))
        ws = torch.index_select(ws, dim, indices)
    assert ws.shape == s_shape
    return ws

def interpolation(wt, s_shape, mode):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    s_len = len(s_shape)

    if s_len == 4:
        ws = ws.permute(3, 2, 1, 0)
        s_shape = s_shape[::-1]
    else:
        s_shape = torch.Size([1, s_shape[0]]) if s_len == 1 else s_shape
        while len(ws.shape) < 4:
            ws = ws.unsqueeze(0)

    ws = torch.nn.functional.interpolate(ws, size=s_shape[-2:], mode='bilinear', align_corners=None)

    if s_len == 4:
        ws = ws.permute(3, 2, 1, 0)
    else:
        while len(ws.shape) > s_len:
            ws = ws.squeeze(0)    
    return ws

def pooling(wt, s_shape):
    assert wt.dim() == len(s_shape), "Tensors have different number of dimensions"
    ws = wt.clone()
    for dim in range(wt.dim()):
        # determine whether teacher is larger than student on this dimension
        assert wt.shape[dim] >= s_shape[dim], "Teacher's dimension should not be smaller than student's dimension"
        # ws = ...
    assert ws.shape == s_shape
    return ws


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.model_type == 'vit':
        teacher = timm.create_model(args.pretrained_model, pretrained=True)
        student = vit_tiny()
    elif args.model_type == 'convnext':
        teacher = timm.create_model(args.pretrained_model, pretrained=True)
        student = convnext_femto()
    else:
        raise ValueError("Invalid model type specified.")

    teacher_weights = teacher.state_dict()
    student_weights = student.state_dict()

    if args.init == 'uniform':
        init_fn = uniform_element_selection
    elif args.init in ['bilinear', 'nearest']:
        init_fn = lambda w, s: interpolation(w, s, args.init)
    elif args.init == 'pooling':
        init_fn = pooling
    else:
        raise ValueError("Invalid initialization method specified.")

    weight_selection = {}
    for key in student_weights.keys():
        if "head" in key:
            continue
        weight_selection[key] = init_fn(teacher_weights[key], student_weights[key].shape)

    if args.output_dir.endswith(".pt") or args.output_dir.endswith(".pth"):
        torch.save(weight_selection, os.path.join(args.output_dir))
    else:
        torch.save(weight_selection, os.path.join(args.output_dir, f"{args.model_type}_{args.init}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for saved model")
    parser.add_argument("--model_type", type=str, default='vit', choices=['vit', 'convnext'], help="Model type: vit or convnext")
    parser.add_argument("--pretrained_model", type=str, default='vit_small_patch16_224_in21k', help="Pretrained model name for timm.create_model")
    parser.add_argument("--init", type=str, default='uniform', choices=['uniform', 'bilinear', 'nearest', 'pooling'], help="Initialization method for student model")
    parser.add_argument("--name", "-n", type=str, default='', help="Device to use for model initialization")

    args = parser.parse_args()
    main(args)