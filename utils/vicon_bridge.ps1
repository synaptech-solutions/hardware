# vicon_bridge.ps1
# ---------------------------------------------------------------------------
# Bridges the Vicon UDP pose broadcast into WSL2.
#
# The Vicon PC (192.168.10.2) broadcasts pose to 255.255.255.255:51001 at
# ~100 Hz. WSL2 is behind NAT and can't see that broadcast, and Windows itself
# drops it unless this NIC has an address on the 192.168.10.x subnet. This
# script:
#   1. Puts the Ethernet NIC on 192.168.10.x (no gateway/DNS -> internet stays
#      on Wi-Fi).  [persists; needs admin the first time]
#   2. Turns the Public firewall profile OFF while running (a scoped allow rule
#      won't match a limited broadcast) and RESTORES it when the bridge stops.
#   3. Forwards each :51001 datagram to the WSL VM, where rl_dryrun.py listens.
#
# RUN:   powershell -ExecutionPolicy Bypass -File C:\Users\james\Documents\CODE\hardware-windows\vicon_bridge.ps1
#        (Run as Administrator the first time so step 1 can set the IP.)
#        Ctrl+C to stop.
# ---------------------------------------------------------------------------

$ETH     = 'Ethernet'          # NIC connected to the Vicon network
$VSUBNET = '192.168.10.'       # the Vicon subnet (PC is .2)
$MYIP    = '192.168.10.100'    # this laptop's address on that subnet
$PORT    = 51001

# --- 1. make sure the NIC is on the Vicon subnet ---------------------------
$have = Get-NetIPAddress -InterfaceAlias $ETH -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -like "$VSUBNET*" }
if (-not $have) {
    Write-Host ("Ethernet not on {0}x - assigning {1}/24 ..." -f $VSUBNET, $MYIP) -ForegroundColor Yellow
    try {
        Set-NetIPInterface -InterfaceAlias $ETH -Dhcp Disabled -ErrorAction SilentlyContinue
        New-NetIPAddress -InterfaceAlias $ETH -IPAddress $MYIP -PrefixLength 24 -ErrorAction Stop | Out-Null
        Write-Host ("  set {0}/24 (no gateway/DNS - internet stays on Wi-Fi)" -f $MYIP) -ForegroundColor Green
    } catch {
        Write-Host ("  FAILED to set IP: {0}" -f $_.Exception.Message) -ForegroundColor Red
        Write-Host "  -> Re-run this script in an ADMIN PowerShell (right-click > Run as administrator)." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ("Ethernet already on Vicon subnet: {0}" -f $have.IPAddress) -ForegroundColor Green
}

# --- 2. find the WSL VM IP -------------------------------------------------
$wslip = (wsl.exe hostname -I).Trim().Split(' ')[0]
if (-not $wslip) { Write-Host "Could not get WSL IP (is WSL running?)" -ForegroundColor Red; exit 1 }
Write-Host ("Forwarding UDP :{0}  ->  WSL {1}:{0}" -f $PORT, $wslip) -ForegroundColor Cyan

# --- 3. open the sockets ---------------------------------------------------
$recv = New-Object System.Net.Sockets.UdpClient
$recv.Client.SetSocketOption([System.Net.Sockets.SocketOptionLevel]::Socket,
                             [System.Net.Sockets.SocketOptionName]::ReuseAddress, $true)
$recv.EnableBroadcast = $true
$recv.Client.ReceiveTimeout = 1000     # wake ~1/s so Ctrl+C is responsive + cleanup runs
$recv.Client.Bind((New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Any, $PORT)))
$send = New-Object System.Net.Sockets.UdpClient
$dest = New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Parse($wslip), $PORT)

# --- 4. firewall: let the broadcast through while bridging ------------------
# A scoped allow rule won't match a LIMITED broadcast (dst 255.255.255.255), so
# for this isolated lab LAN we just turn the Public firewall profile OFF while
# the bridge runs and RESTORE it on exit. The Vicon net is the only Public
# network, so this only lowers the guard for the duration of the test. (Needs
# admin.) Any stale allow-rule from earlier attempts is removed too.
Remove-NetFirewallRule -DisplayName 'Vicon UDP in' -ErrorAction SilentlyContinue
$fwPrev = $null
try {
    $fwPrev = (Get-NetFirewallProfile -Profile Public).Enabled
    Set-NetFirewallProfile -Profile Public -Enabled False -ErrorAction Stop
    Write-Host "Firewall: Public profile OFF for this session (restored on exit)" -ForegroundColor Yellow
} catch {
    $fwPrev = $null
    Write-Host ("Could not change firewall ({0}) - run as ADMIN." -f $_.Exception.Message) -ForegroundColor Red
}

Write-Host "Bridging... (Ctrl+C to stop)" -ForegroundColor Cyan
$n = 0; $t0 = Get-Date
try {
    while ($true) {
        $ep = New-Object System.Net.IPEndPoint([System.Net.IPAddress]::Any, 0)
        try {
            $data = $recv.Receive([ref]$ep)
        } catch [System.Net.Sockets.SocketException] {
            if ($_.Exception.SocketErrorCode -eq [System.Net.Sockets.SocketError]::TimedOut) { continue }
            throw
        }
        [void]$send.Send($data, $data.Length, $dest)
        $n++
        if (($n % 100) -eq 0) {
            $hz = [math]::Round($n / ((Get-Date) - $t0).TotalSeconds)
            Write-Host ("`r  forwarded {0} pkts (~{1}/s) from {2}      " -f $n, $hz, $ep.Address) -NoNewline
        }
    }
} finally {
    $recv.Close(); $send.Close()
    if ($null -ne $fwPrev) {
        Set-NetFirewallProfile -Profile Public -Enabled $fwPrev -ErrorAction SilentlyContinue
        Write-Host "`nFirewall: Public profile restored."
    }
    Write-Host "stopped."
}
