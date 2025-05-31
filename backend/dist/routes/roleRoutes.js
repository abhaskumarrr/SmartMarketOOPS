"use strict";
/**
 * Role Routes
 * Endpoints for role management and permission operations
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const roleController_1 = require("../controllers/roleController");
const auth_1 = require("../middleware/auth");
const router = express_1.default.Router();
// All routes require authentication
router.use(auth_1.protect);
// Get all roles with permissions
router.get('/', (0, auth_1.requirePermission)(['admin:manageRoles']), roleController_1.getAllRoles);
// Get all available permissions
router.get('/permissions', (0, auth_1.requirePermission)(['admin:manageRoles']), roleController_1.getAllPermissions);
// Check if current user has a specific permission
router.get('/check-permission/:permission', roleController_1.checkPermission);
// Get specific role
router.get('/:roleName', (0, auth_1.requirePermission)(['admin:manageRoles']), roleController_1.getRole);
// Get users by role
router.get('/:roleName/users', (0, auth_1.requirePermission)(['admin:manageRoles', 'users:read']), roleController_1.getUsersByRole);
exports.default = router;
//# sourceMappingURL=roleRoutes.js.map