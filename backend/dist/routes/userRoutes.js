"use strict";
/**
 * User Routes
 * Endpoints for user profile management
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const userController_1 = require("../controllers/userController");
const auth_1 = require("../middleware/auth");
const router = express_1.default.Router();
// Health check for user routes
router.get('/health', (req, res) => {
    res.json({ status: 'User routes working', timestamp: new Date().toISOString() });
});
// Get current user profile
router.get('/profile', auth_1.protect, (0, auth_1.requirePermission)(['profile:read']), userController_1.getProfile);
// Update current user profile
router.put('/profile', auth_1.protect, (0, auth_1.requirePermission)(['profile:update']), userController_1.updateProfile);
// Admin only: Get all users
router.get('/', auth_1.protect, (0, auth_1.requirePermission)(['users:read']), userController_1.getUsers);
// Admin only: Update a user's role
router.put('/:id/role', auth_1.protect, (0, auth_1.requirePermission)(['admin:manageRoles']), userController_1.updateUserRole);
exports.default = router;
//# sourceMappingURL=userRoutes.js.map