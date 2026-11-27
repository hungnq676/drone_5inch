# Thông số drone

Số liệu dùng trong mô hình `mjcf/drone_khung_dac.xml` (khung tấm đặc, bản đang dùng). Mở `mjcf/scene_khung_dac.xml` để xem.
Khối lượng mỗi loại linh kiện khai báo một lần trong `<default>` của file XML; sửa ở đó thì mọi chi tiết cùng loại đổi theo.

## Khối lượng

| Bộ phận | Số lượng | 1 cái | Cộng | Nguồn | Class trong XML |
|---|---|---|---|---|---|
| Tấm đáy 8 mm (PETG-CF, in đặc) | 1 | 94.6 g | 94.6 g | CAD: thể tích × 1.30 g/cm³ | mesh `tam_day` |
| Tấm nóc 4.5 mm | 1 | 22.2 g | 22.2 g | CAD | mesh `tam_noc` |
| Tấm hông phải + trái | 2 | 1.8 g | 3.6 g | CAD | mesh `tam_hong_*_8` |
| Cọc đồng lục giác M3 20 mm | 8 | 4.47 g | 35.8 g | Đồng thau, lục giác 6 mm | `coc20` |
| Ốc M3 10 mm (tấm nóc) | 8 | 1.14 g | 9.1 g | Thép, đầu tròn | `oc10` |
| Ốc M3 18 mm (cọc giữa) | 4 | 1.46 g | 5.8 g | Thép | `oc18` |
| Ốc M3 8 mm (cọc mũi, đuôi) | 4 | 0.91 g | 3.6 g | Thép | `oc8` |
| Ốc M3 20 mm (stack) | 4 | 1.57 g | 6.3 g | Thép | `oc20` |
| Motor DYS Samguk Wei 2207 2600KV | 4 | 34.4 g | 137.6 g | Datasheet: 34.09–34.7 g | `motor` |
| Cánh HQ 5146 (3 lá) | 4 | 4.4 g | 17.6 g | Thông số cánh 5.1 inch 3 lá | `canh` |
| FC MicoAir743v2, 36 × 36 × 8 mm | 1 | 10 g | 10 g | Datasheet | `fc` |
| ESC AM32 50 A 4-in-1, 44 × 43.5 × 5.2 mm | 1 | 14 g | 14 g | Datasheet dòng MicoAir AM32 | `esc` |
| Pin 4S 1300 mAh, 73 × 35 × 36 mm | 1 | 165 g | 165 g | Thông số pin 4S 1300 mAh | `pin` |
| **Tổng** | | | **525 g** | MuJoCo tính lại từ mô hình | |

Chưa đưa vào mô hình: camera FPV, VTX, receiver, dây điện, dây buộc pin.

## Kích thước

| | Giá trị | Nguồn |
|---|---|---|
| Tâm motor trước → tâm motor sau | 134.7 mm | CAD |
| Tâm motor trái → tâm motor phải | 163.8 mm | CAD |
| Khoảng hở đầu cánh trước–sau (cánh 5.1 inch) | khoảng 5 mm | CAD |
| Tấm đáy (gồm cả tay) | 160.7 × 194.1 × 8 mm | CAD |
| Mặt trên tấm đáy / mặt dưới tấm nóc | Z = 2.5 mm / Z = 22.5 mm | CAD |
| ESC / FC trong stack | Z 3.4–8.6 mm / Z 11–19 mm | Bố trí stack 30.5 × 30.5 mm |
| Pin (trên tấm nóc) | Z 27–63 mm | Bố trí |

Gốc toạ độ: tâm 4 motor, Z = 0 cao hơn mặt dưới tấm đáy 5.5 mm. X = mũi, Y = trái, Z = lên.

## Motor + cánh

Motor DYS Wei 2207 2600KV + cánh HQ 5146 + pin 4S. Nguồn: bảng đo lực đẩy Tyto Robotics, pin 4S.

- Lực đẩy lớn nhất: **10.75 N** (1.096 kgf) mỗi motor, ở 22 010 vòng/phút, 32 A
- Tỉ lệ lực đẩy / trọng lượng: 4 × 10.75 / (0.525 × 9.81) = **8.3**
- Hệ số yaw: **k = 0.0123 m** (mô-men xoắn = k × lực đẩy)
- Hệ số lực đẩy: **2.02e-6 N/(rad/s)²** (lực đẩy = hệ số × ω²)

## Chiều quay motor (PX4 Quad X chuẩn, "props in")

Nhìn từ trên xuống, mũi ở phía trên:

| Motor PX4 | Vị trí | Toạ độ (mm) | Chiều quay | Dấu mô-men z trong `gear` |
|---|---|---|---|---|
| M1 | trước phải | (+67.4, −82.0) | ngược chiều kim đồng hồ | **−k** |
| M2 | sau trái | (−67.4, +82.0) | ngược chiều kim đồng hồ | **−k** |
| M3 | trước trái | (+67.4, +81.9) | thuận chiều kim đồng hồ | **+k** |
| M4 | sau phải | (−67.4, −81.9) | thuận chiều kim đồng hồ | **+k** |

"Props in" nghĩa là: ở phía mũi, lá cánh quét **vào giữa thân**.

## Khung cũ (9 chi tiết ghép)

`mjcf/drone.xml` + `mjcf/scene.xml` vẫn giữ mô hình khung gốc "Fpvdrone 5 inch" (Yiğit Can Yıldız): 9 chi tiết PETG-CF (83.3 g),
4 cọc 28 mm + 4 cọc 20 mm, 6 đai ốc rút, tổng 502 g. Khung này được thay bằng khung tấm đặc vì in ra chưa đủ cứng,
danh sách in và lắp khung mới ở [KHUNG_DAC.md](KHUNG_DAC.md).
