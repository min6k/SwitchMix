from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import os

# =========================
# 설정
# =========================
log_path = "/home/mskim/D3T/experiment/flir_rgb2thermal_phat+new/outputs/Switchmix_final6(mask32+fda1.2)_72.07/events.out.tfevents.1762833998.mskim-WS-C621E-SAGE-Series.958615.0"  # 이벤트 파일 경로
save_dir = "/home/mskim/D3T/experiment/flir_rgb2thermal_phat+new/outputs/Switchmix_final6(mask32+fda1.2)_72.07"                          # 저장 폴더
os.makedirs(save_dir, exist_ok=True)

LOSS_TAG = "total_loss"   # ← 실제 tag 이름에 맞게 수정
MAP_TAG  = "bbox/AP50"      # ← 예: val/mAP, detection/mAP 등

# =========================
# 이벤트 로드
# =========================
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()

print("Available scalar tags:")
print(ea.Tags()['scalars'])

# =========================
# Loss 시각화
# =========================
loss_events = ea.Scalars(LOSS_TAG)
loss_steps = [e.step for e in loss_events]
loss_values = [e.value for e in loss_events]

plt.figure(figsize=(7,5))
plt.plot(loss_steps, loss_values, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()

loss_save_path = os.path.join(save_dir, "training_loss.png")
plt.savefig(loss_save_path, dpi=300)
plt.close()

print(f"Loss figure saved to: {loss_save_path}")

# =========================
# mAP 시각화
# =========================
map_events = ea.Scalars(MAP_TAG)
map_steps = [e.step for e in map_events]
map_values = [e.value for e in map_events]

plt.figure(figsize=(7,5))
plt.plot(map_steps, map_values, linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("mAP")
plt.title("Validation mAP")
plt.grid(True)
plt.tight_layout()

map_save_path = os.path.join(save_dir, "validation_map.png")
plt.savefig(map_save_path, dpi=300)
plt.close()

print(f"mAP figure saved to: {map_save_path}")