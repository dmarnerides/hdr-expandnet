from __future__ import division, print_function
import argparse
import os
import numpy as np
import torch
import cv2
from smooth import smoothen_luminance
from model import ExpandNet
from util import (
    process_path,
    split_path,
    map_range,
    str2bool,
    cv2torch,
    torch2cv,
    resize,
    tone_map,
    create_tmo_param_from_args,
)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('ldr', nargs='+', type=process_path, help='Ldr image(s)')
    arg(
        '--out',
        type=lambda x: process_path(x, True),
        default=None,
        help='Output location.',
    )
    arg(
        '--video',
        type=str2bool,
        default=False,
        help='Whether input is a video.',
    )
    arg(
        '--patch_size',
        type=int,
        default=256,
        help='Patch size (to limit memory use).',
    )
    arg('--resize', type=str2bool, default=False, help='Use resized input.')
    arg(
        '--use_exr',
        type=str2bool,
        default=False,
        help='Produce .EXR instead of .HDR files.',
    )
    arg('--width', type=int, default=960, help='Image width resizing.')
    arg('--height', type=int, default=540, help='Image height resizing.')
    arg('--tag', default=None, help='Tag for outputs.')
    arg(
        '--use_gpu',
        type=str2bool,
        default=torch.cuda.is_available(),
        help='Use GPU for prediction.',
    )
    arg(
        '--tone_map',
        choices=['exposure', 'reinhard', 'mantiuk', 'drago', 'durand'],
        default=None,
        help='Tone Map resulting HDR image.',
    )
    arg(
        '--stops',
        type=float,
        default=0.0,
        help='Stops (loosely defined here) for exposure tone mapping.',
    )
    arg(
        '--gamma',
        type=float,
        default=1.0,
        help='Gamma curve value (if tone mapping).',
    )
    arg(
        '--use_weights',
        type=process_path,
        default='weights.pth',
        help='Weights to use for prediction',
    )
    arg(
        '--ldr_extensions',
        nargs='+',
        type=str,
        default=['.jpg', '.jpeg', '.tiff', '.bmp', '.png'],
        help='Allowed LDR image extensions',
    )
    opt = parser.parse_args()
    return opt


def load_pretrained(opt):
    net = ExpandNet()
    net.load_state_dict(
        torch.load(opt.use_weights, map_location=lambda s, l: s)
    )
    net.eval()
    return net


#  def create_preprocess(opt):
#      preprocess = [lambda x: x.astype('float32')]
#      if opt.resize:
#          preprocess.append(partial(resize, size=(opt.width, opt.height)))
#      preprocess.append(map_range)
#      preprocess = compose(preprocess)
#      return preprocess


def preprocess(x, opt):
    x = x.astype('float32')
    if opt.resize:
        x = resize(x, size=(opt.width, opt.height))
    x = map_range(x)
    return x


def create_name(inp, tag, ext, out, extra_tag):
    root, name, _ = split_path(inp)
    if extra_tag is not None:
        tag = '{0}_{1}'.format(tag, extra_tag)
    if out is not None:
        root = out
    return os.path.join(root, '{0}_{1}.{2}'.format(name, tag, ext))


def create_video(opt):
    if opt.tone_map is None:
        opt.tone_map = 'reinhard'
    net = load_pretrained(opt)
    video_file = opt.ldr[0]
    cap_in = cv2.VideoCapture(video_file)
    fps = cap_in.get(cv2.CAP_PROP_FPS)
    width = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #  preprocess = create_preprocess(opt)
    n_frames = cap_in.get(cv2.CAP_PROP_FRAME_COUNT)
    predictions = []
    lum_percs = []
    while cap_in.isOpened():
        perc = cap_in.get(cv2.CAP_PROP_POS_FRAMES) * 100 / n_frames
        print('\rConverting video: {0:.2f}%'.format(perc), end='')
        ret, loaded = cap_in.read()
        if loaded is None:
            break
        ldr_input = preprocess(loaded, opt)
        t_input = cv2torch(ldr_input)
        if opt.use_gpu:
            net.cuda()
            t_input = t_input.cuda()
        predictions.append(
            torch2cv(net.predict(t_input, opt.patch_size).cpu())
        )
        percs = np.percentile(predictions[-1], (1, 25, 50, 75, 99))
        lum_percs.append(percs)
    print()
    cap_in.release()

    smooth_predictions = smoothen_luminance(predictions, lum_percs)
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out_vid_name = create_name(
        video_file, 'prediction', 'avi', opt.out, opt.tag
    )
    out_vid = cv2.VideoWriter(out_vid_name, fourcc, fps, (width, height))
    for i, pred in enumerate(smooth_predictions):
        perc = (i + 1) * 100 / n_frames
        print('\rWriting video: {0:.2f}%'.format(perc), end='')
        tmo_img = tone_map(
            pred, opt.tone_map, **create_tmo_param_from_args(opt)
        )
        tmo_img = (tmo_img * 255).astype(np.uint8)
        out_vid.write(tmo_img)
    print()
    out_vid.release()


def create_images(opt):
    #  preprocess = create_preprocess(opt)
    net = load_pretrained(opt)
    if (len(opt.ldr) == 1) and os.path.isdir(opt.ldr[0]):
        # Treat this as a directory of ldr images
        opt.ldr = [
            os.path.join(opt.ldr[0], f)
            for f in os.listdir(opt.ldr[0])
            if any(f.lower().endswith(x) for x in opt.ldr_extensions)
        ]
    for ldr_file in opt.ldr:
        loaded = cv2.imread(
            ldr_file, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        )
        if loaded is None:
            print('Could not load {0}'.format(ldr_file))
            continue
        ldr_input = preprocess(loaded, opt)
        if opt.resize:
            out_name = create_name(
                ldr_file, 'resized', 'jpg', opt.out, opt.tag
            )
            cv2.imwrite(out_name, (ldr_input * 255).astype(int))

        t_input = cv2torch(ldr_input)
        if opt.use_gpu:
            net.cuda()
            t_input = t_input.cuda()
        prediction = map_range(
            torch2cv(net.predict(t_input, opt.patch_size).cpu()), 0, 1
        )

        extension = 'exr' if opt.use_exr else 'hdr'
        out_name = create_name(
            ldr_file, 'prediction', extension, opt.out, opt.tag
        )
        print(f'Writing {out_name}')
        cv2.imwrite(out_name, prediction)
        if opt.tone_map is not None:
            tmo_img = tone_map(
                prediction, opt.tone_map, **create_tmo_param_from_args(opt)
            )
            out_name = create_name(
                ldr_file,
                'prediction_{0}'.format(opt.tone_map),
                'jpg',
                opt.out,
                opt.tag,
            )
            cv2.imwrite(out_name, (tmo_img * 255).astype(int))


def main():
    opt = get_args()
    if opt.video:
        create_video(opt)
    else:
        create_images(opt)


if __name__ == '__main__':
    main()
