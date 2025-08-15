use burn::tensor::backend::Backend;
use std::env;

#[derive(Debug, Clone)]
pub struct DataDDP<B: Backend> {
    pub device: B::Device,
    pub ddp_rank: usize,
    pub ddp_local_rank: usize,
    pub ddp_world_size: usize,
    pub master_process: bool,
}

impl<B: Backend> DataDDP<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            ddp_rank: 0,
            ddp_local_rank: 0,
            ddp_world_size: 1,
            master_process: true,
        }
    }

    /// Returns true if DDP is enabled, false otherwise
    pub fn init_ddp(&mut self) -> Result<bool, String> {
        // Check if we're in a distributed environment by looking for RANK environment variable
        let ddp = env::var("RANK").map_or(false, |rank| rank != "-1");

        if ddp {
            // Distributed training setup
            let ddp_rank = env::var("RANK")
                .map_err(|_| "RANK environment variable not set".to_string())?
                .parse::<usize>()
                .map_err(|_| "Invalid RANK value".to_string())?;

            let ddp_local_rank = env::var("LOCAL_RANK")
                .map_err(|_| "LOCAL_RANK environment variable not set".to_string())?
                .parse::<usize>()
                .map_err(|_| "Invalid LOCAL_RANK value".to_string())?;

            let ddp_world_size = env::var("WORLD_SIZE")
                .map_err(|_| "WORLD_SIZE environment variable not set".to_string())?
                .parse::<usize>()
                .map_err(|_| "Invalid WORLD_SIZE value".to_string())?;

            let master_process = ddp_rank == 0;

            self.ddp_rank = ddp_rank;
            self.ddp_local_rank = ddp_local_rank;
            self.ddp_world_size = ddp_world_size;
            self.master_process = master_process;

            println!(
                "Initialized DDP with rank {} of {}",
                ddp_rank, ddp_world_size
            );
            Ok(true)
        } else {
            // Single process training
            self.ddp_rank = 0;
            self.ddp_local_rank = 0;
            self.ddp_world_size = 1;
            self.master_process = true;

            println!("Initialized single process training");
            Ok(false)
        }
    }

    /// Actually create new DataDDP instance for distributed training with device selection
    /// Call after parsing environment variables
    pub fn distributed(
        device: B::Device,
        rank: usize,
        local_rank: usize,
        world_size: usize,
    ) -> Self {
        Self {
            device,
            ddp_rank: rank,
            ddp_local_rank: local_rank,
            ddp_world_size: world_size,
            master_process: rank == 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use std::env;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_new_initialization() {
        let device = Default::default();
        let ddp = DataDDP::<TestBackend>::new(device);

        assert_eq!(ddp.ddp_rank, 0);
        assert_eq!(ddp.ddp_local_rank, 0);
        assert_eq!(ddp.ddp_world_size, 1);
        assert!(ddp.master_process);
    }

    #[test]
    fn test_single_process_init() {
        // Clean environment variables first
        unsafe {
            env::remove_var("RANK");
            env::remove_var("LOCAL_RANK");
            env::remove_var("WORLD_SIZE");
        }

        let device = Default::default();
        let mut ddp = DataDDP::<TestBackend>::new(device);

        let result = ddp.init_ddp();
        assert!(result.is_ok());
        assert!(!result.unwrap()); // Should return false for single process
        assert!(ddp.master_process);
        assert_eq!(ddp.ddp_rank, 0);
        assert_eq!(ddp.ddp_world_size, 1);
    }

    #[test]
    fn test_distributed_init() {
        // Clean environment variables first
        unsafe {
            env::remove_var("RANK");
            env::remove_var("LOCAL_RANK");
            env::remove_var("WORLD_SIZE");
        }

        let device = Default::default();
        let mut ddp = DataDDP::<TestBackend>::new(device);

        // Set up DDP environment variables
        unsafe {
            env::set_var("RANK", "1");
            env::set_var("LOCAL_RANK", "1");
            env::set_var("WORLD_SIZE", "4");
        }

        let result = ddp.init_ddp();
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should return true for distributed
        assert!(!ddp.master_process); // Rank 1 is not master
        assert_eq!(ddp.ddp_rank, 1);
        assert_eq!(ddp.ddp_local_rank, 1);
        assert_eq!(ddp.ddp_world_size, 4);

        // Clean up
        unsafe {
            env::remove_var("RANK");
            env::remove_var("LOCAL_RANK");
            env::remove_var("WORLD_SIZE");
        }
    }

    #[test]
    fn test_device_access() {
        let device = Default::default();
        let ddp = DataDDP::<TestBackend>::new(device);

        // Should be able to get a reference to the device
        let _device_ref = ddp.device;
    }
}
