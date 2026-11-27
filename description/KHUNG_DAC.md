# Khung 5 inch bản tấm đặc: danh sách để in và lắp

Các file in nằm trong description/meshes/ (đơn vị mm, đặt sẵn tư thế in; khác các STL gốc tính bằng mét, đặt đúng vị trí lắp).

## In (PETG-CF, infill 100%, không cần support)
| File | Số lượng | Đặt khi in | Khoảng |
|---|---|---|---|
| tam_day_8mm.stl | 1 | mặt phẳng úp xuống (đúng như file) | 95 g |
| tam_noc_4.5mm.stl | 1 | mặt có hốc ngàm úp xuống (đúng như file) | 22 g |
| tam_hong_phai_8mm.stl | 1 | nằm nghiêng (mặt 3 mm úp xuống) | 1.8 g |
| tam_hong_trai_8mm.stl | 1 | nằm nghiêng (mặt 3 mm úp xuống) | 1.8 g |

## Cọc và ốc
| Món | Số lượng | Chỗ | So với khung gốc |
|---|---|---|---|
| Cọc đồng lục giác M3 **20 mm** | **8** | 4 cọc giữa + 2 mũi + 2 đuôi | gốc: 4 × 20 mm + 4 × 28 mm |
| Ốc M3 **10 mm** | 8 | tấm nóc -> đầu trên 8 cọc | gốc: M3 8 mm (tấm nóc dày hơn 2 mm) |
| Ốc M3 18 mm | 4 | mặt dưới tấm đáy -> 4 cọc giữa | như gốc |
| Ốc M3 8 mm | 4 | mặt dưới tấm đáy (trong hốc) -> 4 cọc mũi/đuôi | như gốc |
| Ốc M3 20 mm + đai ốc | 4 | stack ESC + FC (đầu ốc nằm trong hốc ở mặt dưới) | như gốc |
| Ốc motor M3 | 16 | 4 mỗi motor (đầu ốc nằm trong hốc, còn 5.5 mm nhựa) | như gốc |

Không cần nữa: 6 đai ốc rút M3, 6 ốc M3 11 mm (trước dùng để ghép tấm đáy + tay + tấm kẹp).

## Khác khung gốc khi lắp
- Motor ngồi cao hơn 2.5 mm, ốc motor vẫn dùng loại cũ.
- Chỗ cho camera giữa 2 tấm hông còn 20 mm cao (gốc 28 mm): ướm thử camera trước.
- Mạch VTX/RX ở đuôi (lỗ M2) ngồi cao hơn 8 mm so với gốc.
