import org.eclipse.paho.client.mqttv3.*;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MqttConfig {

    @Bean
    public IMqttClient mqttClient() throws MqttException {
        String broker = "tcp://YOUR_RASPBERRY_PI_IP:1883";
        String clientId = MqttClient.generateClientId();
        IMqttClient mqttClient = new MqttClient(broker, clientId);
        MqttConnectOptions options = new MqttConnectOptions();
        options.setAutomaticReconnect(true);
        options.setCleanSession(true);
        mqttClient.connect(options);
        return mqttClient;
    }
}
