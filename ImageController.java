import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class ImageController {

    @Autowired
    private MqttService mqttService;

    @GetMapping("/")
    public String index(Model model) {
        return "index";
    }

    @GetMapping("/image")
    @ResponseBody
    public byte[] getImage() {
        return mqttService.getImageData();
    }
}
