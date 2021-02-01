import matplotlib.pyplot as plt
import json 
import os
import numpy as np
import seaborn as sns
import pickle 


def plot_loss(save_folder, txt_log):
    style = 'dark'
    sns.set_style(style)
    out = os.path.join(save_folder, 'loss.png')
    title = None
    train_stats = {'Iter': [], 'loss': [], 'LR': []}
    with open(txt_log) as f:
        for line in f.readlines():
            if 'Iter' not in line:
                continue
            line = line.strip('\n').split(' || ')
            train_stats['Iter'].append(int(line[2].split(' ')[1].split('/')[0]))
            train_stats['loss'].append(float(line[3].split(' ')[-1]))
            # train_stats['LR'].append(float(line[4].split(' ')[-1]))
    metrics = ['loss']
    legend  = ['loss']

    i = 0
    num_metrics = len(metrics)
    iters = train_stats['Iter']
    for j, metric in enumerate(metrics):
        xs  = np.asarray(iters)
        ys = train_stats[metric]
        plt.xlabel('Iter')
        plt.plot(xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)

        plt.legend()
    if title is not None:
        plt.title(title)
    
    print(f'save curve to: {out}')
    plt.savefig(out)
    plt.show()
    plt.cla()


def plot_map(save_folder, ap_stats, metrics, legend, fig_name):
    style = 'dark'
    sns.set_style(style)
    out = os.path.join(save_folder, fig_name)
    title = None

    i = 0
    num_metrics = len(metrics)
    epochs = ap_stats['epoch']
    for j, metric in enumerate(metrics):
        xs  = np.asarray(epochs)
        ys = ap_stats[metric]
        ax = plt.gca()
        ax.set_xticks(xs)
        plt.xlabel('epoch')
        plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')

        plt.legend()
    if title is not None:
        plt.title(title)
    
    print(f'save curve to: {out}')
    plt.savefig(out)
    plt.show()
    plt.cla()


if __name__ == '__main__':
    # save_folder = os.path.join('eval/', 'lr_5e4')

    # # ap_stats = {'ap50': [0.6415997014050593, 0.5897942969887763, 0.6015789946337639, 0.6510466277942631, 0.6795438518773913, 0.657016347661626, 0.634813413413633, 0.6736749100778101, 0.6384532836680722, 0.6398784071599725, 0.6429058473359669, 0.6811069509114638, 0.6782105507667507, 0.6778297150399736, 0.678461698697665, 0.6853066015111229, 0.6846410551498128, 0.6793285875173107, 0.6819632029754589, 0.684359500571572, 0.6851529061645676], 'ap_small': [0.07030197668431065, 0.04438532090466426, 0.07089373029387948, 0.04970344295030928, 0.08467318501671922, 0.05578963383112475, 0.05589996721934994, 0.06067379925071138, 0.07465161830032152, 0.06168251624839604, 0.07202808160242785, 0.06679151051246718, 0.0709071951474324, 0.07121706257803022, 0.06689025219392394, 0.057350187688843035, 0.07241314586553176, 0.06336797370667113, 0.06558225661434755, 0.06694407974341818, 0.0673558810299327], 'ap_medium': [0.3215823461657895, 0.309562648026919, 0.2840365778743828, 0.34971281621116973, 0.34968580686578266, 0.2942963648608278, 0.33213591253490043, 0.3610534741645778, 0.31312731967874463, 0.3421530380998552, 0.29204082550509247, 0.3523960810266547, 0.33777173821174783, 0.34021187990084384, 0.33618205071590496, 0.3397393034849228, 0.34174686029922635, 0.3452845276559563, 0.34876546324274116, 0.34677440097427864, 0.34469121736332514], 'ap_large': [0.20093391509971492, 0.26135052768659445, 0.19554826405925246, 0.24661499553727878, 0.3364588657227521, 0.2510942011939022, 0.2582452320649261, 0.22351731421986012, 0.31327941857294833, 0.2676041856941739, 0.24077684105028507, 0.3086235799012604, 0.29220169117980305, 0.3126918658988431, 0.2930001743432865, 0.3047871277453702, 0.3129423522142375, 0.30918007673941966, 0.3109696368230692, 0.3037086545173769, 0.30883813438445296], 'epoch': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300]}
    # # res_file = None
    
    # res_file = os.path.join(save_folder, 'ap_stats.json')
    # if res_file:
    #     # print('Writing ap stats json to {}'.format(res_file))
    #     # with open(res_file, 'w') as fid:
    #     #     json.dump(ap_stats, fid)
    #     with open(res_file) as f:
    #         ap_stats = json.load(f)

    # metrics = ['ap', 'ap50', 'ap_small', 'ap_medium', 'ap_large']
    # legend  = ['ap', 'ap50', 'ap_small', 'ap_medium', 'ap_large']
    # plot_map(save_folder, ap_stats, metrics, legend)

    save_folder = os.path.join('eval/', 'solo_cs_fcos_2e3')
    coco_eval_pkl = os.path.join(save_folder, 'detection_results.pkl')
    
    f = open(coco_eval_pkl,'rb')
    coco_eval = pickle.load(f)

    # extract eval data
    precisions = coco_eval.eval["precision"]
    '''
    precisions[T, R, K, A, M]
    T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
    R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
    K: category, idx from 0 to ...
    A: area range, (all, small, medium, large), idx from 0 to 3 
    M: max dets, (1, 10, 100), idx from 0 to 2
    '''
    pr_array1 = precisions[0, :, 0, 0, 2] 
    pr_array2 = precisions[1, :, 0, 0, 2] 
    pr_array3 = precisions[2, :, 0, 0, 2] 
    pr_array4 = precisions[3, :, 0, 0, 2] 
    pr_array5 = precisions[4, :, 0, 0, 2] 
    pr_array6 = precisions[5, :, 0, 0, 2] 
    pr_array7 = precisions[6, :, 0, 0, 2] 
    pr_array8 = precisions[7, :, 0, 0, 2] 
    pr_array9 = precisions[8, :, 0, 0, 2] 
    pr_array10 = precisions[9, :, 0, 0, 2] 

    x = np.arange(0.0, 1.01, 0.01)
    # plot PR curve
    plt.plot(x, pr_array1, label="iou=0.5")
    plt.plot(x, pr_array2, label="iou=0.55")
    plt.plot(x, pr_array3, label="iou=0.6")
    plt.plot(x, pr_array4, label="iou=0.65")
    plt.plot(x, pr_array5, label="iou=0.7")
    plt.plot(x, pr_array6, label="iou=0.75")
    plt.plot(x, pr_array7, label="iou=0.8")
    plt.plot(x, pr_array8, label="iou=0.85")
    plt.plot(x, pr_array9, label="iou=0.9")
    plt.plot(x, pr_array10, label="iou=0.95")

    plt.xlabel("recall")
    plt.ylabel("precison")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.show()