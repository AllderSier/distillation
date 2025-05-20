import os

from .train_teachers import main as train_teachers_main
from .train_student_baseline import main as train_baseline_main
from .train_student_kd_ce import main as train_student_kd_ce_main
from .train_student_feature_kd_ce import main as train_feature_kd_ce_main
from .train_student_multi_teacher_response_kd_ce import main as train_mt_resp_main
from .train_student_multi_teacher_feature_kd_ce import main as train_mt_feat_main
from .train_multi_pupil_majority_vote import main as train_mp_logits_main
from .train_multi_pupil_majority_vote_feature import main as train_mp_feat_main
from .train_data_free_random_gen import main as train_df_rand_logits_main
from .train_data_free_random_DI import main as train_df_rand_feat_main

def main():
    print("===== 0) Teachers =====")
    if not (os.path.exists("teacher_custom.pth")
            and os.path.exists("teacher_resnet16.pth")
            and os.path.exists("teacher_vgg16.pth")):
        train_teachers_main()
    else:
        print("Teachers already present — skip.\n")

    experiments = [
        ("Baseline student",                      train_baseline_main            ),
        ("KD + CE",                               train_student_kd_ce_main       ),
        ("Feature-KD + CE",                       train_feature_kd_ce_main       ),
        ("Multi-teacher (logits) + CE",           train_mt_resp_main             ),
        ("Multi-teacher (features) + CE",         train_mt_feat_main             ),
        ("Multi-pupil ensemble (logits)",         train_mp_logits_main           ),
        ("Multi-pupil ensemble (features)",       train_mp_feat_main             ),
        ("Data-free (random generator logits)",   train_df_rand_logits_main      ),
        ("Data-free (random generator features)", train_df_rand_feat_main        ),
    ]

    for idx, (title, func) in enumerate(experiments, start=1):
        print(f"\n===== {idx}) {title} =====")
        func()

    print("\n===== DONE: all experiments finished. =====")

if __name__ == "__main__":
    main()