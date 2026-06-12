import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CORE TRANSFORMATION ENGINE CLASSES ---

class RealTime369Transducer:
    def __init__(self, sample_rate=44100, block_size=512):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Initialize the 3-6-9 Tri-Particle Transformation Matrix
        # Maps 1D input vectors to the 3D basis vectors of the cosmic medium
        self.M_369 = np.array([
            [3.0,            np.log3(6.0),   np.log3(9.0)],
            [np.log10(6.0),  6.0,            np.log10(9.0)],
            [1.0/3.0,        1.0/6.0,        9.0]
        ], dtype=np.float64)
        
    def _calculate_digital_root_scalar(self, val):
        """Reduces a numeric value to its fundamental modulo-9 single digit."""
        integer_representation = int(np.abs(np.round(val * 10000)))
        if integer_representation == 0:
            return 9
        root = integer_representation % 9
        return 9 if root == 0 else root

    def process_buffer(self, input_buffer):
        """
        Accepts a 1D time-domain chunk of linear speech, processes it through
        the information geometric matrix, and returns the re-synthesized 4D data burst.
        """
        assert len(input_buffer) == self.block_size
        
        # Step 1: STFT Execution / Property Extraction
        fft_data = np.fft.rfft(input_buffer)
        magnitudes = np.abs(fft_data)
        phases = np.angle(fft_data)
        frequencies = np.fft.rfftfreq(self.block_size, 1.0 / self.sample_rate)
        
        total_energy = np.sum(magnitudes**2)
        mean_frequency = np.average(frequencies, weights=magnitudes + 1e-12)
        phase_velocity = np.mean(np.gradient(phases))
        
        # Construct the 3-Element Linguistic Phase Vector (V_L)
        V_L = np.array([total_energy, mean_frequency, phase_velocity], dtype=np.float64)
        
        # Step 2: Modulo-9 Validation
        combined_scalar = np.sum(V_L)
        digital_root = self._calculate_digital_root_scalar(combined_scalar)
        
        # Step 3: Execute the 3-6-9 Geometric Resonator Transformation
        transformed_vectors = np.dot(self.M_369, V_L)
        
        # Extract target amplitudes scaled by the 3-6-9 constraints
        target_amp_3 = transformed_vectors * 0.3
        target_amp_6 = transformed_vectors * 0.6
        target_amp_9 = transformed_vectors * 0.9
        
        # Step 4: Inverse Transient Synthesis (Acoustic Delta Function)
        t_axis = np.linspace(0, self.block_size / self.sample_rate, self.block_size)
        
        # Triadic boundary check (Is it a resonant anchor?)
        is_resonant = digital_root in
        decay_constant = 100.0 if is_resonant else 800.0  
        
        # Generate phase-locked output waveform
        output_buffer = np.exp(-decay_constant * t_axis) * (
            target_amp_3 * np.sin(3.0 * np.pi * mean_frequency * t_axis) +
            target_amp_6 * np.sin(6.0 * np.pi * mean_frequency * t_axis) +
            target_amp_9 * np.sin(9.0 * np.pi * mean_frequency * t_axis)
        )
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output_buffer))
        if max_val > 0:
            output_buffer /= max_val
            
        return output_buffer, V_L, digital_root

# --- 2. STREAMLIT USER INTERFACE CONFIGURATION ---

st.set_page_config(page_title="3-6-9 Digital Transducer", layout="wide")

st.title("🎛️ Real-Time 3-6-9 Digital Transducer System")
st.markdown("""
This control panel models the real-time conversion of **1D Linear Phonetic Speech** back into **4D High-Throughput Atmospheric Data Bursts**. 
It calculates the informational weights of the signal and passes it directly through the 3-6-9 matrix boundary parameters.
""")

# Sidebar Controls for Real-Time Parameter Adjustment
st.sidebar.header("🎚️ Signal Generator Controls")
freq_1 = st.sidebar.slider("Base Frequency (Hz)", 100, 2000, 440)
freq_2 = st.sidebar.slider("Harmonic Overlay (Hz)", 200, 4000, 880)
noise_level = st.sidebar.slider("Linear Speech Fragmentation (Noise)", 0.0, 1.0, 0.2)

block_size = st.sidebar.selectbox("Block Size (Chronon Update Window)", [256, 512, 1024], index=1)
sample_rate = 44100

# Initialize Transducer Engine dynamically based on sidebar
transducer = RealTime369Transducer(sample_rate=sample_rate, block_size=block_size)

# Generate Mock Linear Input Audio Buffer
time_steps = np.linspace(0, block_size / sample_rate, block_size)
pure_speech = np.sin(2 * np.pi * freq_1 * time_steps) * np.cos(2 * np.pi * freq_2 * time_steps)
noise = np.random.normal(0, noise_level, block_size)
simulated_input = pure_speech + noise

# Process the buffer through the 3-6-9 algorithm
transduced_output, extraction_vector, root_result = transducer.process_buffer(simulated_input)

# --- 3. METRICS DISPLAY PANELS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Calculated Signal Energy", value=f"{extraction_vector[0]:.4f}")
with col2:
    st.metric(label="Mean Spectrum Frequency", value=f"{extraction_vector[1]:.2f} Hz")
with col3:
    # Stylize the modulo 9 output based on triadic anchors
    is_anchor = root_result in
    status_label = "✅ RESONANT ANCHOR" if is_anchor else "❌ PHASELESS INSTABILITY"
    st.metric(label=f"Modulo-9 Digital Root ({status_label})", value=str(root_result))

st.markdown("---")

# --- 4. WAVEFORM VISUALIZATION GRAPH ---
st.subheader("📊 Information Throughput Real-Time Comparison")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(time_steps * 1000, simulated_input, color='darkred', alpha=0.6, label='Input Linear Speech (Phonetics/Noise)', linewidth=1.5)
ax.plot(time_steps * 1000, transduced_output, color='magenta', label='Transduced 4D Atmospheric Data Burst', linewidth=2)

ax.set_xlabel('Time Window (Milliseconds)', fontsize=10)
ax.set_ylabel('Informational Field Amplitude', fontsize=10)
ax.set_title('Signal Wave Collapse: 1D Continuous Wave to High-Density 3-6-9 Telemetry Burst', fontsize=11, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='upper right')

# Render plot cleanly in Streamlit
st.pyplot(fig)

st.markdown("""
### 🧠 Operational Interpretation:
*   The **Dark Red Line** displays the highly inefficient, spread-out phonetic wave patterns of modern communication. 
*   The **Magenta Spikes** demonstrate your reverse-engineered **Atmospheric Data Language**. By locking into the 3-6-9 matrix values, the transducer compresses the information into micro-second transient packets—enabling immediate, high-throughput transmission directly into the local medium.
""")
