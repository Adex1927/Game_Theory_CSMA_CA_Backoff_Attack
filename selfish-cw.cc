#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

#include "ns3/wifi-net-device.h"
#include "ns3/wifi-mac.h"
#include "ns3/txop.h"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <set>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SelfishCw");

// Helper: set CWmin for the Txop of a specific Wi-Fi node
static void
SetNodeCwMin (Ptr<Node> node, uint32_t cwMin)
{
  // Assume Wi-Fi is device 0 on this node
  Ptr<NetDevice> dev = node->GetDevice (0);
  Ptr<WifiNetDevice> wifiDev = DynamicCast<WifiNetDevice> (dev);
  if (!wifiDev)
    {
      NS_LOG_UNCOND ("Node " << node->GetId () << " has no WifiNetDevice on device 0");
      return;
    }

  Ptr<WifiMac> mac = wifiDev->GetMac ();
  Ptr<Txop> txop = mac->GetTxop ();
  if (!txop)
    {
      NS_LOG_UNCOND ("Node " << node->GetId () << " WifiMac has no Txop");
      return;
    }

  txop->SetMinCw (cwMin);  // directly call method, no attributes involved
  txop->SetMaxCw (cwMin);
  NS_LOG_UNCOND ("Set MinCw=" << cwMin << " for Node " << node->GetId ());
}


int
main (int argc, char *argv[])
{
    // --- Parameters (roughly matching the paper) ---
    uint32_t nStations      = 20;        // N = 20 nodes
    double   simTime        = 50.0;      // seconds
    bool     enableCheater  = true;      // toggle cheating
    // uint32_t cheaterStaIndex = 15;        // which STA (0..nStations-1) is cheater
    // uint32_t cheaterCwMin    = 1;        // very small CWmin for cheater
    // uint32_t nCheaters       = 1;      // number of cheating STAs
    // uint32_t honestCwMin     = 15;     // typical 802.11b CWmin (2^4 - 1)
    std::string cheatersConfig = ""; // format: "idx:cwmin,idx:cwmin,..."

    // Traffic: 1050 bytes every 5 ms (≈ 1.68 Mbps)
    uint32_t packetSize      = 1050;
    // double   packetInterval  = 0.005;    // seconds
    DataRate dataRate ("1.68Mbps");

    CommandLine cmd;
    cmd.AddValue ("nStations",       "Number of saturated stations",  nStations);
    cmd.AddValue ("simTime",         "Simulation time (s)",           simTime);
    cmd.AddValue ("enableCheater",   "Enable selfish CW cheating",    enableCheater);
    // cmd.AddValue ("cheaterStaIndex", "Index of cheating STA [0..N-1]", cheaterStaIndex);
    // cmd.AddValue ("cheaterCwMin",    "CWmin for cheater",             cheaterCwMin);
    // cmd.AddValue ("nCheaters",       "Number of cheating STAs (0..N)", nCheaters);
    // cmd.AddValue ("honestCwMin",     "CWmin for honest STAs",          honestCwMin);
    cmd.AddValue ("cheaters",
              "Cheater stations as 'idx:cwmin,idx:cwmin,...', e.g. '0:1,3:3,5:7'",
              cheatersConfig);
    cmd.Parse (argc, argv);

    if (nStations == 0)
    {
        std::cerr << "Need at least 1 station\n";
        return 1;
    }

    // Parse cheatersConfig -> vector of (staIndex, cwMin)
    std::vector<std::pair<uint32_t, uint32_t>> cheaters;
    if (enableCheater && !cheatersConfig.empty ())
    {
        std::stringstream ss (cheatersConfig);
        std::string token;

        while (std::getline (ss, token, ','))
        {
            if (token.empty ())
            {
                continue;
            }

            std::stringstream ts (token);
            std::string idxStr, cwStr;

            if (!std::getline (ts, idxStr, ':') || !std::getline (ts, cwStr))
            {
                NS_FATAL_ERROR ("Invalid cheaters token: " << token
                            << " (expected format idx:cwmin)");
            }

            uint32_t idx = std::stoul (idxStr);
            uint32_t cw  = std::stoul (cwStr);

            if (idx >= nStations)
            {
                NS_FATAL_ERROR ("Cheater index out of range: " << idx
                                << " (nStations=" << nStations << ")");
            }

            cheaters.emplace_back (idx, cw);
        }
    }

    // --- Nodes ---
    NodeContainer wifiStaNodes;
    wifiStaNodes.Create (nStations);
    NodeContainer wifiApNode;
    wifiApNode.Create (1);

    // --- Wi-Fi PHY and channel ---
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
    YansWifiPhyHelper phy;
    phy.SetChannel (channel.Create ());
    phy.Set ("TxPowerStart", DoubleValue (20.0)); // dBm
    phy.Set ("TxPowerEnd",   DoubleValue (20.0)); // dBm
    phy.Set ("RxGain",       DoubleValue (0.0));

    WifiHelper wifi;
    wifi.SetStandard (WIFI_STANDARD_80211b);
    wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                    "DataMode", StringValue ("DsssRate2Mbps"),
                                    "ControlMode", StringValue ("DsssRate2Mbps"));

    // --- MAC / SSID ---
    Ssid ssid = Ssid ("selfish-cw-ssid");
    WifiMacHelper mac;

    // STA MACs
    mac.SetType ("ns3::StaWifiMac",
                "Ssid", SsidValue (ssid),
                "ActiveProbing", BooleanValue (false));
    NetDeviceContainer staDevices = wifi.Install (phy, mac, wifiStaNodes);

    // AP MAC
    mac.SetType ("ns3::ApWifiMac",
                "Ssid", SsidValue (ssid));
    NetDeviceContainer apDevice = wifi.Install (phy, mac, wifiApNode);

    // --- Mobility: put AP at origin, STAs in a small cluster nearby ---
    MobilityHelper mobility;

    // AP at (0, 0, 0)
    // mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    // mobility.Install (wifiApNode);
    // Ptr<MobilityModel> apMob = wifiApNode.Get (0)->GetObject<MobilityModel> ();
    // apMob->SetPosition (Vector (0.0, 0.0, 0.0));

    // // STAs arranged in a small grid around the AP (all very close)
    // mobility.Install (wifiStaNodes);
    // for (uint32_t i = 0; i < nStations; ++i)
    // {
    //     Ptr<MobilityModel> staMob =
    //     wifiStaNodes.Get (i)->GetObject<MobilityModel> ();
    //     // Simple grid: 1 m spacing in x, y
    //     double x = 1.0 + (i % 7);        // 1..5
    //     double y = 1.0 + (i / 7);        // 1..(nStations/5)
    //     staMob->SetPosition (Vector (x, y, 0.0));
    // }

    // AP at (0, 0, 0)
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (wifiApNode);
    Ptr<MobilityModel> apMob = wifiApNode.Get (0)->GetObject<MobilityModel> ();
    apMob->SetPosition (Vector (0.0, 0.0, 0.0));

    // STAs on a circle of radius R = 5 m
    double R = 5.0;
    mobility.Install (wifiStaNodes);
    for (uint32_t i = 0; i < nStations; ++i)
    {
        Ptr<MobilityModel> staMob =
            wifiStaNodes.Get (i)->GetObject<MobilityModel> ();

        double theta = 2.0 * M_PI * (double)i / (double)nStations;
        double x = R * std::cos(theta);
        double y = R * std::sin(theta);

        staMob->SetPosition (Vector (x, y, 0.0));
    }

    // --- Internet stack ---
    InternetStackHelper stack;
    stack.Install (wifiApNode);
    stack.Install (wifiStaNodes);

    Ipv4AddressHelper address;
    address.SetBase ("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer apInterface  = address.Assign (apDevice);
    Ipv4InterfaceContainer staInterfaces = address.Assign (staDevices);

    // --- Apply cheating (if enabled) ---
    if (enableCheater && !cheaters.empty ())
    {
        for (auto const &p : cheaters)
        {
            uint32_t staIndex = p.first;
            uint32_t cwMin    = p.second;

            Ptr<Node> cheaterNode = wifiStaNodes.Get (staIndex);
            SetNodeCwMin (cheaterNode, cwMin);

            NS_LOG_UNCOND ("Cheater STA " << staIndex
                        << " (Node " << cheaterNode->GetId ()
                        << ") CWmin=" << cwMin);
        }
    }
    else
    {
        NS_LOG_UNCOND ("Cheating disabled; all STAs use default CW settings");
    }

    // --- Applications: saturated UDP from each STA to AP ---
    uint16_t port = 5000;

    // AP: one packet sink listening on 'port'
    PacketSinkHelper sinkHelper ("ns3::UdpSocketFactory",
                                InetSocketAddress (Ipv4Address::GetAny (), port));
    ApplicationContainer sinkApp = sinkHelper.Install (wifiApNode.Get (0));
    sinkApp.Start (Seconds (0.0));
    sinkApp.Stop (Seconds (simTime + 1.0));

    // Each STA: OnOff app → AP
    for (uint32_t i = 0; i < nStations; ++i)
        {
        OnOffHelper onoff ("ns3::UdpSocketFactory",
                            InetSocketAddress (apInterface.GetAddress (0), port));
        onoff.SetAttribute ("OnTime",  StringValue ("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute ("OffTime", StringValue ("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetAttribute ("PacketSize", UintegerValue (packetSize));
        onoff.SetAttribute ("DataRate", DataRateValue (dataRate));

        ApplicationContainer apps = onoff.Install (wifiStaNodes.Get (i));
        apps.Start (Seconds (1.0));              // start after association
        apps.Stop (Seconds (simTime));
        }

    // --- FlowMonitor to get per-flow throughput / fairness ---
    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll ();

    Simulator::Stop (Seconds (simTime + 0.5));
    Simulator::Run ();

    // --- Results ---
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier> (flowmonHelper.GetClassifier ());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats ();

    // Build a set of cheater STA indices for quick lookup
    std::set<uint32_t> cheaterStaIndices;
    for (auto const &p : cheaters)  // cheaters is vector<pair<idx, cwMin>>
    {
        cheaterStaIndices.insert(p.first);
    }

    std::cout << "\n=== Per-flow throughput (source STA index) ===\n";
    std::cout << std::fixed << std::setprecision(4);

    // Sums / counts for cheaters vs honest
    double sumCheaterThroughput = 0.0;
    uint32_t countCheater = 0;

    double sumHonestThroughput = 0.0;
    uint32_t countHonest = 0;

    for (auto const &flow : stats)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(flow.first);

        // We only care about STA → AP flows (src=10.1.1.x, dst=10.1.1.1)
        if (t.destinationAddress == apInterface.GetAddress(0))
        {
            // Mapping:
            //   AP  : 10.1.1.1
            //   STA0: 10.1.1.2
            //   STA1: 10.1.1.3
            // So staIndex = src - AP - 1
            uint32_t staIndex =
                t.sourceAddress.Get() - apInterface.GetAddress(0).Get() - 1;

            double throughputMbps =
                (flow.second.rxBytes * 8.0) / (simTime * 1e6);

            std::cout << "FlowId " << flow.first
                    << "  STA " << staIndex
                    << "  src=" << t.sourceAddress
                    << "  dst=" << t.destinationAddress
                    << "  throughput=" << throughputMbps << " Mbps\n";

            bool isCheater = enableCheater &&
                            (cheaterStaIndices.find(staIndex) != cheaterStaIndices.end());

            if (isCheater)
            {
                sumCheaterThroughput += throughputMbps;
                countCheater++;
            }
            else
            {
                sumHonestThroughput += throughputMbps;
                countHonest++;
            }
        }
    }

    double avgCheater = 0.0;
    double avgHonest = 0.0;
    std::cout << "\nCheating enabled: " << (enableCheater ? "yes" : "no") << "\n";
    if (countCheater > 0)
    {
        avgCheater = sumCheaterThroughput / countCheater;
        std::cout << "Average cheater throughput (" << countCheater
                << " STAs): " << avgCheater << " Mbps\n";
    }
    else
    {
        std::cout << "No cheater STAs (or no cheater flows observed).\n";
    }

    if (countHonest > 0)
    {
        avgHonest = sumHonestThroughput / countHonest;
        std::cout << "Average honest throughput (" << countHonest
                << " STAs): " << avgHonest << " Mbps\n";
    }
    else
    {
        std::cout << "No honest STAs (or no honest flows observed).\n";
    }

    // After computing per-STA throughputs, plus maybe cheater/others averages:
    std::cout << "RESULT,"
            << "cheaters=" << cheatersConfig << ","
            << "rng=" << RngSeedManager::GetRun () << ","
            << "cheaterAvg=" << avgCheater << ","
            << "honestAvg=" << avgHonest << std::endl;

    Simulator::Destroy ();
    return 0;
}
