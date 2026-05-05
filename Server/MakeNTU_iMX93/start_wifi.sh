modprobe moal mod_para=nxp/wifi_mod_para.conf
ifconfig mlan0 up
ifconfig mlan0
killall wpa_supplicant 2>/dev/null
ip link set mlan0 up
wpa_supplicant -B -i mlan0 -D nl80211 -c /etc/wpa_supplicant.conf
udhcpc -i mlan0

echo "nameserver 1.1.1.1" > /etc/resolv.conf
