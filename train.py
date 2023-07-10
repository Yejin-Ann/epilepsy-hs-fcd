""" Training script for MS lesion generation and pseudo-healthy synthesis

You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train:
        python3 train.py --dataroot ./datasets/healthy2pathological --name h2p_version0 --model agan_foreground


See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_nifti, test_saving, save_nifti_ver2
from util.visualizer import Visualizer
from util.util import save_image_as_nifti
import numpy as np



if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks_3D; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    result_dir = opt.results_dir # yb
    
    for epoch in range(opt.epoch_count, opt.epoch + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        # by yb
        A_20 = dataset.dataset.A_paths[:20]
        B_20 = dataset.dataset.B_paths[:20]
        A_list = []
        B_list = []
        for i in range(len(A_20)):
            subid_A = A_20[i].split('/')[-1][6:8]
            subid_B = B_20[i].split('/')[-1][5:7]
        
            A_list.append(subid_A)
            B_list.append(subid_B)
        # --------------------------------------------

        for i, data in enumerate(dataset):  # inner loop within one epoch

            # test
            test_saving(data['A'], 'data_from_dataset')

            iter_start_time = time.time()  # timer for computation per iteration
            # by yb
            if data['A'].size()[0]==2:
                healthy_1 = data['A_paths'][0].split('/')[-1][6:8]
                healthy_2 = data['A_paths'][1].split('/')[-1][6:8]
                pathology_1 = data['B_paths'][0].split('/')[-1][5:7]
                pathology_2 = data['B_paths'][1].split('/')[-1][5:7]
            else:
                healthy_1 = data['A_paths'][0].split('/')[-1][6:8]
                pathology_1 = data['B_paths'][0].split('/')[-1][5:7]
            # lesion_subid = data['C_paths'][0].split('/')[-1][:7]
            # if batch_size changed, need to be changed --------------------------------
            
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            #     save_result = total_iters % opt.update_html_freq == 0
            #     model.compute_visuals() # nothing
            #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result) # save as image

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks_3D(save_suffix)

            # # by yb
            visuals = model.get_current_visuals()
            if data['A'].size()[0]==2:
                if healthy_1 in A_list:
                    subid = 'subid_'+healthy_1
                    fake_data = visuals['fake_B'][0]
                    save_nifti_ver2('fake_B', fake_data, result_dir, subid, epoch)
                    rec_data = visuals['rec_A'][0]
                    save_nifti_ver2('rec_A', rec_data, result_dir, subid, epoch)
                    A_list.remove(healthy_1)
                if healthy_2 in A_list:
                    subid = 'subid_'+healthy_2
                    fake_data = visuals['fake_B'][1]
                    save_nifti_ver2('fake_B', fake_data, result_dir, subid, epoch)
                    rec_data = visuals['rec_A'][1]
                    save_nifti_ver2('rec_A', rec_data, result_dir, subid, epoch)
                    A_list.remove(healthy_2)
                
                if pathology_1 in B_list:
                    subid = 'subid_'+pathology_1
                    fake_data = visuals['fake_A'][0]
                    save_nifti_ver2('fake_A', fake_data, result_dir, subid, epoch)
                    rec_data = visuals['rec_B'][0]
                    save_nifti_ver2('rec_B', rec_data, result_dir, subid, epoch)
                # lesion_data = visuals['fake_C'][0]
                # save_nifti_ver2('fake_C', lesion_data, result_dir, subid, epoch)
                    B_list.remove(pathology_1)
                if pathology_2 in B_list:
                    subid = 'subid_'+pathology_2
                    fake_data = visuals['fake_A'][1]
                    save_nifti_ver2('fake_A', fake_data, result_dir, subid, epoch)
                    rec_data = visuals['rec_B'][1]
                    save_nifti_ver2('rec_B', rec_data, result_dir, subid, epoch)
                # lesion_data = visuals['fake_C'][1]
                # save_nifti_ver2('fake_C', lesion_data, result_dir, subid, epoch)
                    B_list.remove(pathology_2)
            else:
                if healthy_1 in A_list:
                    subid = 'subid_'+healthy_1
                    fake_data = visuals['fake_B'][0]
                    save_nifti_ver2('fake_B', fake_data, result_dir, subid, epoch)
                    rec_data = visuals['rec_A'][0]
                    save_nifti_ver2('rec_A', rec_data, result_dir, subid, epoch)
                    A_list.remove(healthy_1)
                if pathology_1 in B_list:
                    subid = 'subid_'+pathology_1
                    fake_data = visuals['fake_A'][0]
                    save_nifti_ver2('fake_A', fake_data, result_dir, subid, epoch)
                    rec_data = visuals['rec_B'][0]
                    B_list.remove(pathology_1)
            #------------------------------------------------------

            iter_data_time = time.time()
        
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks_3D('latest')
            model.save_networks_3D(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.epoch, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
    
        # visuals = model.get_current_visuals()
        # result_dir = opt.results_dir
        #save_nifti(visuals, result_dir, epoch)