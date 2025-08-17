#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Label Generation Test Script (English Version)
Demonstrates how to use auto label generation for online learning
"""

import time
import numpy as np
from pretrain_online_learning import PretrainOnlineLearning

def simulate_eeg_data():
    """Simulate EEG data"""
    # Simulate 16 channels, 2 seconds data, 500Hz sampling rate
    channels = 16
    duration = 2
    fs = 500
    samples = duration * fs
    
    # Generate random EEG data
    data = np.random.randn(channels, samples) * 0.1
    
    # Add some periodic signals to simulate brain waves
    t = np.linspace(0, duration, samples)
    for ch in range(channels):
        # Add alpha waves (8-13 Hz)
        alpha_freq = np.random.uniform(8, 13)
        data[ch] += 0.05 * np.sin(2 * np.pi * alpha_freq * t)
        
        # Add beta waves (13-30 Hz)
        beta_freq = np.random.uniform(13, 30)
        data[ch] += 0.03 * np.sin(2 * np.pi * beta_freq * t)
    
    return data

class MockReceiver:
    """Mock data receiver"""
    def __init__(self):
        self.buffer = None
        self.update_buffer()
    
    def update_buffer(self):
        """Update buffer data"""
        # Simulate real-time data stream
        new_data = simulate_eeg_data()
        if self.buffer is None:
            self.buffer = new_data
        else:
            # Add new data to buffer end
            self.buffer = np.concatenate([self.buffer, new_data], axis=1)
            # Keep buffer size (keep last 10 seconds data)
            max_samples = 10 * 500  # 10 seconds * 500Hz
            if self.buffer.shape[1] > max_samples:
                self.buffer = self.buffer[:, -max_samples:]
    
    def get_buffer_data(self, data_type='processed'):
        """Get buffer data"""
        self.update_buffer()
        return self.buffer

def test_auto_label_generation():
    """Test auto label generation functionality"""
    print("🧪 Starting Auto Label Generation Test")
    print("="*60)
    
    # Create mock receiver
    mock_receiver = MockReceiver()
    
    # Create online learning manager
    learning_manager = PretrainOnlineLearning(receiver=mock_receiver)
    
    # Ensure auto label mode is enabled
    learning_manager.set_auto_label_mode(True)
    
    print("✅ Initialization completed")
    print("📋 Test Instructions:")
    print("   - System will automatically generate a new target label every 5 seconds")
    print("   - Label 7 means 'Left hand imagination', Label 8 means 'Right hand imagination'")
    print("   - You have 2 seconds preparation time, then system collects 2 seconds EEG data for learning")
    print("   - Press Ctrl+C to stop test")
    print("="*60)
    
    try:
        # Start online learning
        learning_manager.start_online_learning()
        
        # Run for a period
        test_duration = 60  # Test for 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            time.sleep(1)
            
            # Display current status
            status = learning_manager.get_current_status()
            if status['is_running']:
                print(f"⏱️ Runtime: {time.time() - start_time:.1f}s, "
                      f"Predictions: {status['total_predictions']}, "
                      f"Accuracy: {status['current_accuracy']:.3f}")
        
        print("\n✅ Test completed")
        
    except KeyboardInterrupt:
        print("\n🛑 User interrupted test")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        # Stop online learning
        learning_manager.stop_online_learning()
        
        # Display final results
        stats = learning_manager.get_prediction_statistics()
        print("\n" + "="*60)
        print("📊 Final Test Results")
        print("="*60)
        print(f"Total predictions: {stats['total_predictions']}")
        print(f"Correct predictions: {stats['correct_predictions']}")
        print(f"Final accuracy: {stats['accuracy']:.3f}")
        print("="*60)

if __name__ == "__main__":
    test_auto_label_generation() 