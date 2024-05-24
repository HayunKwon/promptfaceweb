package com.example.mqtt_webapp;

import org.eclipse.paho.client.mqttv3.IMqttClient;
import org.eclipse.paho.client.mqttv3.MqttException;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;

@Service
public class MqttService {

    private final IMqttClient mqttClient;
    private byte[] imageData;

    public MqttService(IMqttClient mqttClient) {
        this.mqttClient = mqttClient;
    }

    @PostConstruct
    public void init() throws MqttException {
        mqttClient.subscribe("home/stream", (topic, message) -> {
            imageData = message.getPayload();
        });
    }

    public byte[] getImageData() {
        return imageData;
    }
}
