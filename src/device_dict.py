class DeviceDict(dict):
    def to(self, device):
        return {k: v.to(device) for k, v in self.items()}
