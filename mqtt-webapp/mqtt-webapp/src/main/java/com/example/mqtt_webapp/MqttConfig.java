package com.example.mqtt_webapp;

import org.eclipse.paho.client.mqttv3.MqttClient;
import org.eclipse.paho.client.mqttv3.MqttConnectOptions;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.eclipse.paho.client.mqttv3.IMqttClient;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MqttConfig {

    @Bean
    public IMqttClient mqttClient() throws MqttException {
        String broker = "tcp://YOUR_RASPBERRY_PI_IP:1883"; // 여기에 실제 라즈베리 파이 IP를 입력하세요
        String clientId = MqttClient.generateClientId();
        IMqttClient mqttClient = new MqttClient(broker, clientId);
        MqttConnectOptions options = new MqttConnectOptions();
        options.setAutomaticReconnect(true);
        options.setCleanSession(true);
        mqttClient.connect(options);
        return mqttClient;
    }
}
