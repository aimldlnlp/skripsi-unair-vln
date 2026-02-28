import argparse
import os
from task_gen import gen_task
from dataset_gen import gen_traj
from split_task import split_traj, gen_step_task


nav_gen_path = os.getcwd()
project_path = os.path.dirname(nav_gen_path)

if not os.path.exists(nav_gen_path + '/task'):    # for LH-VLN task
    os.makedirs(nav_gen_path + '/task')
if not os.path.exists(nav_gen_path + '/step_task'):    # for step-by-step task
    os.makedirs(nav_gen_path + '/step_task')
if not os.path.exists(nav_gen_path + '/logs'):    # for logs
    os.makedirs(nav_gen_path + '/logs')

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--API_KEY', type=str, help="api key for gpt")
    parser.add_argument('--API_BASE', type=str,
        default=os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        help='OpenAI-compatible base URL (contoh: http://localhost:8000/v1, http://localhost:11434/v1)')

    parser.add_argument('--MODEL_MAIN', type=str,
        default=os.environ.get('NAVGEN_MODEL_MAIN', 'gpt-4o'),
        help='Model untuk forward generation (task-level)')

    parser.add_argument('--MODEL_MINI', type=str,
        default=os.environ.get('NAVGEN_MODEL_MINI', 'gpt-4o-mini'),
        help='Model untuk backward/step-by-step generation')

    parser.add_argument('--MODEL_VISION', type=str,
        default=os.environ.get('NAVGEN_MODEL_VISION', 'gpt-4o'),
        help='Model untuk vision (opsional; Qwen 7B Instruct = teks saja)')

    parser.add_argument('--TEMPERATURE', type=float, default=0.2,
        help='Suhu sampling LLM (semakin rendah, semakin deterministik)')

    parser.add_argument('--MAX_TOKENS', type=int, default=1024,
        help='Batas token keluaran per call')

    # generate task
    parser.add_argument('--scene_path', type=str, default=nav_gen_path + '/scene/', help="root scene path")
    parser.add_argument('--prompt_path', type=str, default=nav_gen_path + '/prompt/', help="root prompt path")
    parser.add_argument('--task_path', type=str, default=nav_gen_path + '/task/', help="root task path")
    parser.add_argument('--region_file', type=str, default=nav_gen_path + '/scene/Per_Scene_Region_Weighted_Votes.csv', help="region semantic file")
    parser.add_argument('--loop', type=int, default=100, help="num of tasks to generate")
    
    parser.add_argument('--scene_id', type=str, default=None, help="Whether or not to select a specific scene")
    parser.add_argument('--sample_region', type=bool, default=False, help="Whether or not to sample room in a scene")
    parser.add_argument('--sample_obj', type=bool, default=True, help="Whether or not to sample obj in a room")

    # simulation
    parser.add_argument('--scene', type=str, default=project_path + '/data/hm3d/', help="scene path")
    parser.add_argument('--scene_dataset', type=str, default=project_path + '/data/hm3d/hm3d_annotated_basis.scene_dataset_config.json', help="scene dataset path")
    parser.add_argument('--max_step', type=int, default=500, help="max step for record")
    parser.add_argument('--success_dis', type=float, default=1, help="distance to be considered as success")

    # step
    parser.add_argument('--step_task_path', type=str, default=nav_gen_path + '/step_task/', help="root task path")
    parser.add_argument('--ram_model', type=str, default=project_path + '/data/models/ram_plus_swin_large_14m.pth', help="path for ram model")
    parser.add_argument('--ram_logs', type=str, default=nav_gen_path + '/logs/step_task_logs.txt', help="path to save logs")
    parser.add_argument('--split_save_path', type=str, default=nav_gen_path + '/task/trail_list.txt', help="path to save the split trajectory")


    args = parser.parse_args()
    return args


def main():
    args = read_args()

    # step 1: Generate task
    for i in range(args.loop):
        gen_task(args)

    # step 2: Generate trajectory
    gen_traj(args)

    # step 3: Split trajectory into segments
    split_traj(args)

    # step 4: Generate step-by-step task
    gen_step_task(args)

if __name__ == '__main__':
    main()
