import os
import torch

def save_checkpoint(model_vae, optimizer, global_step, parameter_name, output_dir, logger, ppl=False, ddpm=None, use_philly=False):
    # Create output directory if needed
    # Save model checkpoint
    save_last = 1
    model_to_save = model_vae.module if hasattr(model_vae,
                                                'module') else model_vae  # Take care of distributed/parallel training
    ddpm_to_save = ddpm.module if hasattr(ddpm, 'module') else ddpm
    state_dict_new = {}
    state_dict = model_to_save.state_dict()
    ddpm_state_dict = ddpm_to_save.state_dict()
    for key in parameter_name:
        if key in state_dict.keys():
            state_dict_new[key] = state_dict[key]

    checkpoint = {
        'iter': global_step,
        'model_state_dict': state_dict_new,  # model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'beta': model_to_save.args.beta,
    }
    checkpoint_ddpm = {
        'iter': global_step,
        'model_state_dict': ddpm_state_dict,
    }
    if ppl:
        save_last = 2
    output_ddpm_dir = os.path.join(output_dir, 'checkpoint-ddpm-{}'.format(save_last))
    #if not os.path.exists(output_ddpm_dir) and local_rank in [-1, 0]:?
    if not os.path.exists(output_ddpm_dir):
        os.makedirs(output_ddpm_dir)
    torch.save(checkpoint_ddpm, os.path.join(output_ddpm_dir, 'training_ddpm.bin'))
    logger.info("Saving DDPM checkpoint to %s", output_ddpm_dir)
    
    output_full_dir = os.path.join(output_dir, 'checkpoint-full-{}'.format(save_last))
    # if not os.path.exists(output_full_dir) and local_rank in [-1, 0]:
    if not os.path.exists(output_full_dir):
        os.makedirs(output_full_dir)

    logger.info("Start saving full model checkpoint to %s", output_full_dir)
    if use_philly:
        save_solid = False
        n_save_attempts = 0
        while not save_solid:
            try:
                n_save_attempts += 1
                logger.info(f"Saving full checkpoint: {n_save_attempts} attempts made")
                torch.save(checkpoint, os.path.join(output_full_dir, 'training.bin'))
                logger.info("Saving full checkpoint to %s,", output_full_dir)
                save_solid = True
            except:
                pass
    else:
        torch.save(checkpoint, os.path.join(output_full_dir, 'training.bin'))
        logger.info("Saving full checkpoint to %s", output_full_dir)