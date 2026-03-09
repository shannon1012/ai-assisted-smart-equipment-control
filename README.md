# AI-Assisted Smart Equipment Control

An industrial equipment control system built on ISA-95 architecture principles, integrating physical simulation, anomaly detection, and AI-driven decision making.

Developed in collaboration with **Marketech International** for deployment in **TSMC** semiconductor fabrication facilities.

---

## Use Case: Semiconductor Chiller System (半導體冷卻系統)

Semiconductor fabs require ultra-precise temperature control to maintain yield. This project simulates a **Chiller** — the cooling unit responsible for regulating coolant temperature supplied to process tools (e.g., CVD, CMP, etching chambers).

The simulator models key chiller dynamics including:

- Coolant supply/return temperature
- Compressor load and COP (Coefficient of Performance)
- Flow rate and pressure
- Fault conditions (e.g., refrigerant leak, pump failure, fouling)

This simulation provides a realistic data source for developing and validating the anomaly detection and AI agent layers before deployment on actual fab equipment.

---

## System Architecture

The system is organized into three layers following the ISA-95 industrial automation standard:

```
┌─────────────────────────────────────────────────────┐
│         Level 3: Cognitive / Agent Layer            │
│              AI Agent  &  Reporter                  │
├─────────────────────────────────────────────────────┤
│         Level 2: Monitoring & Detection Layer       │
│        Anomaly Detection  &  Streamlit Dashboard    │
├─────────────────────────────────────────────────────┤
│         Level 1: Physical / Control Layer           │
│           Simulator  ↔  PID / Rule Controller       │
└─────────────────────────────────────────────────────┘
```

### Level 1 — Physical / Control Layer (控制層)

Handles real-time equipment simulation and low-level control logic.

- **Simulator**: Emulates physical equipment behavior and generates sensor data
- **PID / Rule Controller**: Applies PID control loops and rule-based logic to regulate equipment state

### Level 2 — Monitoring & Detection Layer (監控層)

Observes the control layer and surfaces anomalies and metrics to operators.

- **Anomaly Detection**: Identifies deviations from normal operating conditions using statistical or ML-based methods
- **Streamlit Dashboard**: Provides a real-time visual interface for monitoring equipment status, sensor readings, and alerts

### Level 3 — Cognitive / Agent Layer (認知決策層)

Interprets system state and takes higher-level decisions or generates reports.

- **AI Agent**: Reasons over sensor data and anomaly signals to recommend or execute corrective actions
- **Reporter**: Produces structured summaries and reports of equipment events, anomalies, and agent decisions
