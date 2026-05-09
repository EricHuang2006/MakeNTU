import gpiod
from gpiod.line import Direction, Value

CHIP = "/dev/gpiochip0"

R = 3   # 改成你的 Red GPIO line
G = 2   # 改成你的 Green GPIO line
B = 14  # 改成你的 Blue GPIO line

# 共陽 RGB LED：
# GPIO = 0 -> 亮
# GPIO = 1 -> 滅
ON = Value.INACTIVE    # logical inactive, usually low if not active_low
OFF = Value.ACTIVE     # logical active, usually high if not active_low

config = {
    R: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=OFF),
    G: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=OFF),
    B: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=OFF),
}


def log_gpio_values(lines):
    """
    Debug log for current GPIO logical values.
    This uses the existing requested lines object.
    """
    try:
        r_val = lines.get_value(R)
        g_val = lines.get_value(G)
        b_val = lines.get_value(B)

        print("[GPIO DEBUG]")
        print(f"  R line {R}: {r_val}")
        print(f"  G line {G}: {g_val}")
        print(f"  B line {B}: {b_val}")

    except Exception as exc:
        print(f"[GPIO DEBUG] Failed to read GPIO values: {exc}")


with gpiod.request_lines(CHIP, consumer="rgb-led", config=config) as lines:
    try:
        print("Initial GPIO values:")
        log_gpio_values(lines)

        while True:
            s = input("輸入 R,G,B 例如 1,0,1：")

            try:
                rv, gv, bv = map(int, s.replace(",", " ").split())
            except ValueError:
                print("輸入格式錯誤，請輸入例如：1,0,1")
                continue

            values = {
                R: ON if rv else OFF,
                G: ON if gv else OFF,
                B: ON if bv else OFF,
            }

            print("[GPIO SET]")
            print(f"  requested R={rv}, G={gv}, B={bv}")
            print(f"  setting values: {values}")

            lines.set_values(values)

            print("After set_values():")
            log_gpio_values(lines)

    except KeyboardInterrupt:
        print("\nTurning LED off...")

        lines.set_values({
            R: OFF,
            G: OFF,
            B: OFF,
        })

        print("Final GPIO values:")
        log_gpio_values(lines)

        print("LED off")
